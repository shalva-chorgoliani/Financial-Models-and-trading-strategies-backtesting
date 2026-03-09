import numpy as np
import pandas as pd
import yfinance as yf
import requests
import matplotlib.pyplot as plt

# ==========================================================
# 1. DATA ACQUISITION & CLEANING
# ==========================================================
def get_data():
    print("Fetching S&P 500 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    sp500 = pd.read_html(html)[0]
    
    # Format tickers for Yahoo Finance (e.g., BRK.B -> BRK-B)
    tickers = sp500["Symbol"].str.replace(".", "-", regex=False).to_list()

    print("Downloading monthly price data (2000-2025)...")
    data = yf.download(
        tickers, 
        start="2000-01-01", 
        end="2025-01-01", 
        interval="1mo", 
        auto_adjust=True, 
        progress=False
    )["Open"]

    # Cleaning: Keep stocks with at least 90% data coverage to avoid survivorship bias artifacts
    # and forward-fill small gaps.
    data = data.dropna(axis=1, thresh=int(len(data) * 0.9)).ffill().dropna()
    return data

prices = get_data()

# ==========================================================
# 2. STRATEGY PARAMETERS
# ==========================================================
FORMATION = 12    # Months to calculate distance
TRADING = 6       # Months to trade the pairs
N_PAIRS = 10      # Number of pairs per sleeve
ENTRY_Z = 2.0     # Z-score to enter trade
EXIT_Z = 0.5      # Z-score to exit trade (partial mean reversion)

# ==========================================================
# 3. MULTI-SLEEVE BACKTEST ENGINE
# ==========================================================
# Create storage for monthly returns for all 6 possible entry months
neutral_sleeves = pd.DataFrame(0.0, index=prices.index, columns=[f"N_Sleeve_{i}" for i in range(TRADING)])
long_only_sleeves = pd.DataFrame(0.0, index=prices.index, columns=[f"L_Sleeve_{i}" for i in range(TRADING)])

price_mat = prices.values
dates = prices.index
n_stocks = price_mat.shape[1]

print(f"Starting backtest on {n_stocks} stocks across {TRADING} staggered sleeves...")

for s in range(TRADING):
    # Each sleeve rebalances every 6 months, starting at its unique offset (0 through 5)
    for t in range(FORMATION + s, len(dates) - TRADING, TRADING):
        
        # --- VECTORIZED FORMATION PERIOD ---
        # Get the 12-month window for all stocks
        form_window = price_mat[t-FORMATION:t, :]
        # Normalize so the start of the window is 1.0
        norm_form = form_window / form_window[0, :]
        
        # Fast SSD using: sum((a-b)^2) = sum(a^2) + sum(b^2) - 2*dot(a, b)
        sq_sums = np.sum(norm_form**2, axis=0)
        dot_prod = np.dot(norm_form.T, norm_form)
        dist_matrix = sq_sums[:, None] + sq_sums[None, :] - 2 * dot_prod
        
        # Get upper triangle to avoid self-pairing and duplicates
        tri_rows, tri_cols = np.triu_indices(n_stocks, k=1)
        best_indices = np.argsort(dist_matrix[tri_rows, tri_cols])[:N_PAIRS]
        
        # --- TRADING PERIOD ---
        trade_window = price_mat[t:t+TRADING, :]
        last_form_px = form_window[-1, :]
        
        for idx in best_indices:
            i, j = tri_rows[idx], tri_cols[idx]
            
            # Normalized prices for the trading window (relative to formation end)
            A_tr = trade_window[:, i] / last_form_px[i]
            B_tr = trade_window[:, j] / last_form_px[j]
            spread = A_tr - B_tr
            
            # Historical Spread Stats
            hist_spread = norm_form[:, i] - norm_form[:, j]
            mu, sigma = np.mean(hist_spread), np.std(hist_spread)
            
            pos = 0 
            entry_A, entry_B, entry_spread = 0, 0, 0
            
            for k in range(len(spread)):
                curr_date = dates[t + k]
                
                if pos == 0:
                    # ENTRY LOGIC
                    if spread[k] > mu + ENTRY_Z * sigma:
                        pos, entry_spread = -1, spread[k]
                        entry_A, entry_B = A_tr[k], B_tr[k]
                    elif spread[k] < mu - ENTRY_Z * sigma:
                        pos, entry_spread = 1, spread[k]
                        entry_A, entry_B = A_tr[k], B_tr[k]
                else:
                    # EXIT & INCREMENTAL PNL LOGIC
                    prev_spread = spread[k-1] if k > 0 else entry_spread
                    prev_A = A_tr[k-1] if k > 0 else entry_A
                    prev_B = B_tr[k-1] if k > 0 else entry_B
                    
                    # 1. Market Neutral PnL contribution
                    pnl_n = pos * (spread[k] - prev_spread)
                    neutral_sleeves.at[curr_date, f"N_Sleeve_{s}"] += (pnl_n / N_PAIRS)
                    
                    # 2. Long-Only PnL contribution
                    # If pos == -1 (Short A, Long B), we take Long B. If pos == 1, Long A.
                    pnl_l = (B_tr[k] - prev_B) if pos == -1 else (A_tr[k] - prev_A)
                    long_only_sleeves.at[curr_date, f"L_Sleeve_{s}"] += (pnl_l / N_PAIRS)
                    
                    if abs(spread[k] - mu) < EXIT_Z * sigma:
                        break

# ==========================================================
# 4. FINAL AVERAGING & PERFORMANCE PLOTTING
# ==========================================================
# Average across sleeves to neutralize start-month bias
neutral_final = neutral_sleeves.mean(axis=1)
long_only_final = long_only_sleeves.mean(axis=1)

def calc_stats(returns):
    total = returns.sum()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(12) if returns.std() != 0 else 0
    return total, sharpe

n_tot, n_sh = calc_stats(neutral_final)
l_tot, l_sh = calc_stats(long_only_final)

print("\n" + "="*55)
print(f"{'Metric':<20} | {'Market Neutral':<15} | {'Long-Only':<15}")
print("-" * 55)
print(f"{'Total Cumulative PnL':<20} | {n_tot:<15.4f} | {l_tot:<15.4f}")
print(f"{'Annualized Sharpe':<20} | {n_sh:<15.2f} | {l_sh:<15.2f}")
print("="*55)



# Visualizing the Sleeves and Averages
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot Neutral Sleeves
for col in neutral_sleeves.columns:
    ax1.plot(neutral_sleeves[col].cumsum(), color='crimson', alpha=0.15, lw=1)
ax1.plot(neutral_final.cumsum(), color='darkred', lw=3, label='Averaged Market Neutral')
ax1.set_title("Market Neutral: Staggered Sleeves (Hedged)")
ax1.legend(); ax1.grid(True, alpha=0.3)

# Plot Long-Only Sleeves
for col in long_only_sleeves.columns:
    ax2.plot(long_only_sleeves[col].cumsum(), color='teal', alpha=0.15, lw=1)
ax2.plot(long_only_final.cumsum(), color='darkslategrey', lw=3, label='Averaged Long-Only')
ax2.set_title("Long-Only: Staggered Sleeves (Unhedged)")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
