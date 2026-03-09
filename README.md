1. Efficient frontier.py

This script constructs optimal portfolios from **S&P 500 stocks** using **GARCH-based return estimation** and the **Elton–Gruber–Padberg (1976) portfolio selection method**.
It downloads monthly price data, estimates expected returns with a **GARCH(1,1) model**, and computes each stock’s **beta** and **idiosyncratic risk** via regression on market returns. 
Stocks are ranked by their **excess return to beta ratio**, and portfolios are formed iteratively according to the Elton–Gruber–Padberg framework.

The script then computes portfolio risk and return using the **covariance matrix**, generates the **efficient frontier**, 
identifies the **maximum Sharpe ratio (tangency) portfolio**, and plots the frontier together with the **Capital Market Line**.

2. Momentum strategy.py

This script **backtests momentum trading strategies** on S&P 500 stocks using monthly data.
It downloads historical prices, computes **log returns**, and calculates momentum as the **rolling sum of past returns** over different lookback periods. 
Each month, stocks are ranked by momentum and portfolios are formed using the **top and bottom 10 percent** of stocks.

The strategy is implemented with **overlapping portfolios**, meaning multiple portfolios are held simultaneously for different holding periods. 
The code tests **16 combinations of lookback and holding horizons** for both **long only** and **long short (market neutral)** strategies.

Finally, it plots the cumulative performance of all strategies and compares them to the **S&P 500 benchmark**.

3. Options Strategy Payoffs.py

This script implements **many common options trading strategies** and visualizes their **payoff profiles at expiration**.

Each function represents a specific strategy (e.g., protective put, straddles, strangles, butterflies, condors, ratio spreads, collars). 
Given option strikes, premiums, and the underlying price range, the code computes **payoffs, break even points, and maximum profit or loss**.

For each strategy, it generates a **payoff diagram** showing how profits change with the underlying asset price at expiry.
This makes it easy to analyze and compare the **risk return characteristics** of different options strategies.

4. Pairs trading strtaegy.py

This script backtests a **pairs trading strategy** on stocks from the **S&P 500**.

It identifies stock pairs with the **smallest historical price distance** over a 12-month formation period, 
then trades them for 6 months when their **spread deviates significantly from the mean (z-score entry/exit rules)**.

The backtest uses **staggered trading sleeves** to avoid timing bias and evaluates both **market-neutral (long–short)** and **long-only** versions, 
reporting **Sharpe ratios and cumulative PnL** and plotting their performance.


6. SMA strategy.py
   
This script backtests a **10-month moving average timing strategy** on the **S&P 500**.

It goes **long when the index is above its 10-month SMA** and moves to **cash (1-month T-bill yield)** when below. Monthly returns are calculated and compared to **buy-and-hold**.

The code then computes **performance metrics** (CAGR, volatility, Sharpe ratio, max drawdown, trades) and plots **cumulative returns and drawdowns** for the strategy vs. the market.
