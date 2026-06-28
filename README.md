# Stock_trade
Investment tracking AI to learn market patterns

Architecture:

    Data Base: SQLite
    - Tickers: Top performer stocks with addition to expand with new IPO on market (SpaceX, etc.)
        Only trade high volume stocks for better predictions
    - yfinance API OHLCV for last 15 years (More than enuogh)
        Fast API to get precise data information
        Schedule daily data save for new day after market close (8pm UK time)
    - For each stock we gather:
        * OHLCV

    Process:
        - open a code aftrer market close. Run to gather data and make calculastions for new positions. Swing trading is best option. a couple of trades per month won t kill profit with fees.