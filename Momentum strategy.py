import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests

#%% 1. Prepare Data
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"} 
html = requests.get(url, headers=headers).text 
sp500 = pd.read_html(html)[0] 
tickers = sp500['Symbol'].str.replace('.', '-', regex=False).to_list() 

start_date = "2000-01-01"
end_date = "2025-01-02"

data = yf.download(tickers, start=start_date, end=end_date, interval="1mo", auto_adjust=True)['Open']
benchmark = yf.download("^GSPC", start=start_date, end=end_date, interval="1mo", auto_adjust=True)['Open']

log_returns = np.log(data).diff()
benchmark_ret = np.log(benchmark).diff().cumsum()

#%% 2. Backtest Engine with Overlapping Portfolios
def run_overlapping_strategy(lb, hp, returns_df, strategy_type='long_only'):
    # Calculate Momentum: rolling sum of log returns
    # We shift(1) to avoid look-ahead bias
    mom = returns_df.rolling(lb).sum().shift(1)
    
    # Generate weights for a NEW portfolio started at each month t
    # This represents ONLY the 1/hp slice of the total portfolio
    new_weights = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)
    
    for date in returns_df.index:
        row = mom.loc[date].dropna()
        if not row.empty:
            k = max(1, int(len(row) * 0.1))
            ranks = row.rank(method='first')
            
            if strategy_type == 'long_only':
                winners = ranks > (len(row) - k)
                new_weights.loc[date, winners[winners].index] = 1.0 / k
            elif strategy_type == 'long_short':
                winners = ranks > (len(row) - k)
                losers = ranks <= k
                new_weights.loc[date, winners[winners].index] = 0.5 / k
                new_weights.loc[date, losers[losers].index] = -0.5 / k

    # Overlapping Logic: The total portfolio weight at time T 
    # is the average of the weights from the last 'hp' formation periods.
    # This simulates holding 'hp' sub-portfolios simultaneously.
    # 
    total_weights = new_weights.rolling(window=hp).mean()
    
    # Calculate returns: apply weights from T-1 to returns at T
    port_ret = (total_weights.shift(1) * returns_df).sum(axis=1)
    return port_ret.cumsum()

#%% 3. Run all 16 Combinations (Long-Only)
lookbacks = [3, 6, 9, 12]
holdings = [3, 6, 9, 12]

results_lo = pd.DataFrame()
results_ls = pd.DataFrame()

for lb in lookbacks:
    for hp in holdings:
        name = f"L{lb}/H{hp}"
        print(f"Calculating {name}...")
        results_lo[name] = run_overlapping_strategy(lb, hp, log_returns, 'long_only')
        results_ls[name] = run_overlapping_strategy(lb, hp, log_returns, 'long_short')

#%% 4. Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
colors = plt.cm.tab20(np.linspace(0, 1, 16))

# Plot 1: Long Only Overlapping
for i, col in enumerate(results_lo.columns):
    ax1.plot(results_lo[col], color=colors[i], alpha=0.7, label=col)
ax1.plot(benchmark_ret, color='black', linewidth=3, linestyle='--', label='S&P 500')
ax1.set_title("16 Overlapping Momentum Strategies: Long-Only", fontsize=14)
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
ax1.grid(True, alpha=0.3)

# Plot 2: Long-Short Overlapping
for i, col in enumerate(results_ls.columns):
    ax2.plot(results_ls[col], color=colors[i], alpha=0.7, label=col)
ax2.plot(benchmark_ret, color='black', linewidth=3, linestyle='--', label='S&P 500')
ax2.set_title("16 Overlapping Momentum Strategies: Long-Short (Market Neutral)", fontsize=14)
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
