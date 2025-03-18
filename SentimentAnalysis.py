import feedparser
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import yfinance as yf
import schedule
import time
import plotly.graph_objects as go
from datetime import datetime
import logging

# Download VADER lexicon if not already present
nltk.download('vader_lexicon')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to fetch news from Economic Times RSS feed
def fetch_et_markets_news(stock_symbol=None):
    url = 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms'
    try:
        feed = feedparser.parse(url)
        news_items = []
        
        # Define keywords based on mode
        if stock_symbol:
            # Broader keywords for stock-specific news
            base_symbol = stock_symbol.upper()
            relevant_keywords = [base_symbol, f"{base_symbol} Industries", "RIL"]  # e.g., RELIANCE, Reliance Industries, RIL
            mode = f"Stock: {base_symbol}"
            fallback_keywords = ['NSE', 'BSE', 'Sensex', 'Nifty']  # Fallback for market context
        else:
            relevant_keywords = ['NSE', 'BSE', 'Indian stock market', 'Sensex', 'Nifty']
            mode = "Market"
            fallback_keywords = None
        
        # Log raw feed entries for debugging
        logging.debug(f"Raw feed entries: {len(feed['entries'])} items")
        for i, entry in enumerate(feed['entries']):
            logging.debug(f"Entry {i}: {entry.get('title', 'No title')} - {entry.get('description', 'No desc')}")

        # Primary filtering
        for entry in feed['entries']:
            title = entry.get('title', '').lower()
            description = entry.get('description', '').lower()
            text = title + ' ' + description
            if any(keyword.lower() in text for keyword in relevant_keywords):
                news_items.append({
                    'title': entry.get('title', ''),
                    'description': entry.get('description', ''),
                    'pub_date': entry.get('published', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                })

        # Fallback if no stock-specific news is found
        if not news_items and stock_symbol and fallback_keywords:
            logging.info(f"No {mode}-specific news found. Falling back to market news.")
            for entry in feed['entries']:
                title = entry.get('title', '').lower()
                description = entry.get('description', '').lower()
                text = title + ' ' + description
                if any(keyword.lower() in text for keyword in fallback_keywords):
                    news_items.append({
                        'title': entry.get('title', ''),
                        'description': entry.get('description', ''),
                        'pub_date': entry.get('published', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    })
                    news_items[-1]['note'] = 'Fallback: General market news'

        logging.info(f"Fetched {len(news_items)} relevant news items for {mode}")
        return news_items
    except Exception as e:
        logging.error(f"Error fetching news: {e}")
        return []

# Function to analyze sentiment using VADER
def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    return scores['compound']

# Function to fetch price data using yfinance
def fetch_price_data(stock_symbol=None):
    try:
        if stock_symbol:
            ticker = f"{stock_symbol}.NS"
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1], ticker
            else:
                logging.warning(f"No price data for {ticker}")
                return None, ticker
        else:
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1], "^NSEI"
            else:
                logging.warning("No price data for Nifty 50")
                return None, "^NSEI"
    except Exception as e:
        logging.error(f"Error fetching price data: {e}")
        return None, None

# Main function to process news, fetch price, and visualize
def process_and_visualize(stock_symbol=None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    mode = f"Stock: {stock_symbol}" if stock_symbol else "Market (NSE/BSE)"
    logging.info(f"Processing {mode} at {timestamp}")
    
    # Fetch news
    news_items = fetch_et_markets_news(stock_symbol)
    if not news_items:
        print(f"No news found for {mode} even after fallback.")
        return
    
    # Fetch price data
    current_price, ticker = fetch_price_data(stock_symbol)
    price_str = f"Current Price ({ticker}): {current_price:.2f}" if current_price else f"Price data unavailable for {ticker}"
    print(f"\n{price_str}")
    
    # Process news for sentiment
    sentiment_data = []
    for item in news_items:
        text = item['title'] + ' ' + item['description']
        sentiment = analyze_sentiment(text)
        sentiment_data.append({
            'title': item['title'],
            'sentiment': sentiment,
            'pub_date': item['pub_date'],
            'note': item.get('note', '')
        })
    
    # Create DataFrame
    df = pd.DataFrame(sentiment_data)
    print(f"\nSentiment Analysis Results for {mode}:")
    print(df[['title', 'sentiment', 'note']].to_string(index=False))
    
    # Calculate average sentiment
    avg_sentiment = df['sentiment'].mean()
    sentiment_label = 'Bullish' if avg_sentiment > 0.05 else 'Bearish' if avg_sentiment < -0.05 else 'Neutral'
    print(f"\nAverage Sentiment: {avg_sentiment:.2f} ({sentiment_label})")
    
    # Visualize sentiment and price
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['pub_date'], y=df['sentiment'], name='Sentiment',
        hovertemplate='%{customdata}<br>Sentiment: %{y:.2f}<br>Note: %{text}',
        customdata=df['title'], text=df['note'],
        marker_color=df['sentiment'].apply(lambda x: 'green' if x > 0 else 'red' if x < 0 else 'grey')
    ))
    if current_price:
        fig.add_trace(go.Scatter(
            x=[df['pub_date'].iloc[0], df['pub_date'].iloc[-1]], y=[current_price, current_price],
            mode='lines', name=f'Price ({ticker})', yaxis='y2', line=dict(color='blue', dash='dash')
        ))
    
    fig.update_layout(
        title=f"{mode} Sentiment and Price as of {timestamp}",
        xaxis_title="Publication Date",
        yaxis_title="Sentiment Polarity (-1 to 1)",
        yaxis=dict(range=[-1, 1]),
        yaxis2=dict(title=f"Price ({ticker})", overlaying='y', side='right', showgrid=False) if current_price else None,
        legend=dict(x=0, y=1.1, orientation='h')
    )
    filename = f"sentiment_{stock_symbol if stock_symbol else 'market'}.html"
    fig.write_html(filename)
    print(f"Visualization saved as '{filename}'")

# Function to run the analysis with scheduler
def main(stock_symbol=None):
    process_and_visualize(stock_symbol)
    schedule.every(30).minutes.do(process_and_visualize, stock_symbol=stock_symbol)
    print(f"\nStarting scheduler for {stock_symbol if stock_symbol else 'Market'}. Updates every 30 minutes. Press Ctrl+C to stop.")
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nScheduler stopped.")
            break

# Entry point
if __name__ == "__main__":
    print(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
    main("RELIANCE")  # Run for RELIANCE by default
