# Description:
# This script constructs and analyzes optimal portfolios of S&P 500 stocks using a GARCH-based estimation 
# of expected returns and the portfolio selection framework by Elton, Gruber, and Padberg (1976).
#
# Main steps:
# 1. Downloads monthly price data for all S&P 500 constituents and the market index (^GSPC) via Yahoo Finance.
# 2. Estimates each stock’s expected return maximum likelihood estimation of a GARCH(1,1) model. Variance is estimated using .var()
# 3. Calculates systematic (beta) and unsystematic risk by regressing stock returns on market returns.
# 4. Implements the Elton-Gruber-Padberg portfolio selection method:
#    - Ranks stocks by excess return-to-beta ratio.
#    - Iteratively selects stocks that improve the portfolio’s risk-return tradeoff.
# 5. Computes portfolio return, variance, and standard deviation using the full covariance matrix 
#    (thus accounting for correlations between stocks).
# 6. Generates the efficient frontier for varying risk-free rates and identifies the tangency portfolio
#    (maximum Sharpe ratio portfolio).
# 7. Plots the efficient frontier and Capital Market Line (CML).


import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import requests

#%% Functions

def neg_loglik(params, data, dt):
    mu, omega, p, q = params
    n = len(data)
    if omega <= 0 or p < 0 or q < 0 or p >= 1 or q >= 1:
        return 1e12

    sigma2 = np.empty(n)
    sample_var = np.nanvar(data)
    sigma2[0] = max(sample_var / dt, 1e-8)
    ll = 0.0

    for t in range(n):
        if t > 0:
            resid_prev = data[t-1] - mu*dt
            sigma2[t] = omega + p * sigma2[t-1] + q * (resid_prev**2) / dt
            if sigma2[t] <= 0 or not np.isfinite(sigma2[t]):
                return 1e12
        resid = data[t] - mu*dt
        denom = sigma2[t] * dt
        if denom <= 0 or not np.isfinite(denom):
            return 1e12
        ll += 0.5 * (np.log(2*np.pi) + np.log(denom) + (resid**2) / denom)
    return ll

def fit_garch_mle(log_returns, dt=1/12, start_params=None):
    data = np.asarray(log_returns).astype(float)
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 10:
        raise ValueError("Need more observations")
    mu0 = np.mean(data) / dt
    var0 = np.var(data)
    omega0 = 0.01 * var0
    p0, q0 = 0.85, 0.1
    x0 = np.array([mu0, omega0, p0, q0]) if start_params is None else np.array(start_params)
    bnds = [(None, None), (1e-12, None), (0.0, 0.999), (0.0, 0.999)]

    res = minimize(lambda x: neg_loglik(x, data, dt),
                   x0, method="L-BFGS-B", bounds=bnds,
                   options={"disp": False, "maxiter": 10000})
    est = res.x
    return {
        "mu": est[0],
        "omega": est[1],
        "p": est[2],
        "q": est[3],
        "neg_loglik": float(res.fun),
        "success": res.success,
        "message": res.message,
        "optim_result": res
    }

#%% Get data

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}
html = requests.get(url, headers=headers).text
sp500 = pd.read_html(html)[0]
tickers = sp500['Symbol'].str.replace('.', '-', regex=False).to_list()

data = yf.download(
    tickers=tickers,
    start="1994-01-07",
    end="2025-09-01",
    interval="1mo",
    group_by='ticker',
    auto_adjust=True,
    progress=True
)
monthly_prices = pd.concat(
    {ticker: data[ticker]['Close'] for ticker in tickers if 'Close' in data[ticker]},
    axis=1
)
monthly_returns = monthly_prices.pct_change()

market = yf.download(tickers="^GSPC", start="1994-01-07", end="2025-09-01", interval="1mo")
market_returns = market['Close'].pct_change()

#%% GARCH-based expected returns and variances

mus = {}
for col in monthly_returns.columns:
    series = monthly_returns[col].dropna()
    try:
        fit = fit_garch_mle(series.values, dt=1/12)
        mus[col] = fit["mu"]
    except:
        mus[col] = np.nan

stats = pd.DataFrame({
    'mean': pd.Series(mus),
    'variance': monthly_returns.var() * 12,
    'std_dev': monthly_returns.std() * (12 ** 0.5)
})

#%% Beta and idiosyncratic risk

betas = {}
error_vars = {}
common_index = monthly_returns.index.intersection(market_returns.index)
returns_aligned = monthly_returns.loc[common_index]
RM_aligned = market_returns.loc[common_index]

for ticker in returns_aligned.columns:
    Ri = returns_aligned[ticker].dropna()
    RM = RM_aligned.loc[Ri.index].dropna()
    common_dates = Ri.index.intersection(RM.index)
    Ri = Ri.loc[common_dates]
    RM = RM.loc[common_dates]
    if len(Ri) < 2:
        continue
    X = sm.add_constant(RM)
    model = sm.OLS(Ri, X).fit()
    if len(model.params) < 2:
        continue
    betas[ticker] = model.params[1]
    error_vars[ticker] = np.var(model.resid) * 12

results = pd.DataFrame({
    'beta': pd.Series(betas),
    'error_variance_yearly': pd.Series(error_vars)
})

market_return = fit_garch_mle(market_returns)["mu"]
cov_matrix = monthly_returns.cov() * 12  # annual covariance

#%% Efficient frontier

rfr_range = np.linspace(0.001, 0.20 ,10000)
efficient_frontier = []
all_portfolios = []

for rfr in rfr_range:
    if abs(rfr - market_return) < 0.005:
        continue

    temp_stats = stats.copy()
    temp_stats['excess_return'] = temp_stats['mean'] - rfr
    temp_stats['beta'] = results['beta']
    temp_stats['unsystematic_risk'] = results['error_variance_yearly']
    temp_stats = temp_stats[
        (temp_stats['beta'] > 0.1) &
        (temp_stats['beta'] < 5) &
        (temp_stats['unsystematic_risk'] > 0.00001) &
        (temp_stats['unsystematic_risk'] < 100) &
        (temp_stats['mean'] < 0.5) &
        (abs(temp_stats['excess_return']) > 0.00001)
    ]
    if len(temp_stats) < 2:
        continue

    temp_stats['excess_return_over_beta'] = temp_stats['excess_return'] / temp_stats['beta']
    temp_stats = temp_stats.sort_values('excess_return_over_beta', ascending=False)

    temp_stats['step_3'] = (temp_stats['excess_return'] * temp_stats['beta']) / temp_stats['unsystematic_risk']
    temp_stats['step_4'] = (temp_stats['beta']**2) / temp_stats['unsystematic_risk']
    temp_stats['step_5'] = temp_stats['step_3'].cumsum()
    temp_stats['step_6'] = temp_stats['step_4'].cumsum()
    temp_stats['C'] = (cov_matrix.mean().mean() * temp_stats['step_5']) / (1 + cov_matrix.mean().mean() * temp_stats['step_6'])

    portfolio = temp_stats[temp_stats['excess_return_over_beta'] > (temp_stats['C'] + 1e-6)]
    if len(portfolio) < 1:
        continue

    C_star = portfolio['C'].max()
    portfolio['Z'] = (portfolio['beta'] / portfolio['unsystematic_risk']) * (
        (portfolio['excess_return'] / portfolio['beta']) - C_star)
    if portfolio['Z'].abs().sum() < 1e-6:
        continue
    portfolio['weight'] = portfolio['Z'] / portfolio['Z'].sum()

    weights = portfolio['weight'].values
    port_idx = portfolio.index
    port_cov = cov_matrix.loc[port_idx, port_idx].values
    portfolio_variance = np.dot(weights.T, np.dot(port_cov, weights))
    portfolio_std_dev = np.sqrt(portfolio_variance)
    portfolio_return = (portfolio['weight'] * portfolio['mean']).sum()

    if portfolio_std_dev < 1:
        efficient_frontier.append((portfolio_std_dev, portfolio_return))
        all_portfolios.append({
            'rfr': rfr,
            'std_dev': portfolio_std_dev,
            'return': portfolio_return,
            'weights': portfolio['weight'].to_dict()
        })

efficient_frontier = pd.DataFrame(efficient_frontier, columns=['Standard Deviation', 'Return'])
efficient_frontier = efficient_frontier.drop_duplicates().sort_values('Standard Deviation')
detailed_portfolios = pd.DataFrame(all_portfolios)


#%% Step 6: Plot

rf_current = 0.0368  # in decimal form (3.68%)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    efficient_frontier['Standard Deviation'],
    efficient_frontier['Return'],
    c=np.linspace(0, 1, len(efficient_frontier)),
    cmap='viridis',
    s=50,
    alpha=0.7
)
plt.xlabel('Portfolio Standard Deviation')
plt.ylabel('Portfolio Return')
plt.title('Optimized Efficient Frontier (Filtered Stocks)')
plt.grid(True)
plt.ylim(bottom=0)
plt.xlim(left=0)

# Plot a single Capital Market Line for rf_current
# First find the “best” portfolio in your frontier relative to rf_current (max Sharpe ratio)
sharpe = (efficient_frontier['Return'] - rf_current) / efficient_frontier['Standard Deviation']
best_idx = sharpe.idxmax()
best = efficient_frontier.loc[best_idx]

port_risk = best['Standard Deviation']
port_return = best['Return']

# slope = (return - rf) / risk
slope = (port_return - rf_current) / port_risk

# Draw line from zero to beyond the portfolio risk
x_vals = np.array([0, port_risk * 1.2])
y_vals = rf_current + slope * x_vals
plt.plot(x_vals, y_vals, 'r--', linewidth=1.5, label=f'CML @ rf = {rf_current:.2%}')

# Highlight the best Sharpe portfolio
plt.scatter(port_risk, port_return, color='gold', s=200, marker='*', label='Market Portfolio')

plt.legend()
plt.show()


