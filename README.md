# Offline Stock Sentiment Analysis

A command-line tool to analyze stock market sentiment for NSE/BSE stocks using news from Economic Times and price data from Yahoo Finance. Runs entirely offline after fetching data, with results displayed in the terminal and saved locally.

## Features
- **User Input**: Enter an NSE stock symbol (e.g., "RELIANCE") via the command line.
- **Sentiment Analysis**: Uses NLTK VADER to evaluate news sentiment (Bullish, Bearish, Neutral).
- **Price Data**: Fetches latest closing prices via `yfinance`.
- **Offline Output**: Displays results in the terminal, plots a chart with `matplotlib`, and saves to files.
- **No Web Dependency**: Fully functional without a browser or server.

## Prerequisites
- Python 3.10+
- Internet connection (for initial data fetch only)








