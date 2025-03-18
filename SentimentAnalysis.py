import feedparser
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER
sid = SentimentIntensityAnalyzer()

# Fetch news from Economic Times RSS feed
def fetch_et_markets_news(stock_symbol=None):
    url = 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms'
    try:
        feed = feedparser.parse(url)
        news_items = []
        if stock_symbol:
            base_symbol = stock_symbol.upper()
            relevant_keywords = [base_symbol, f"{base_symbol} Industries", "RIL" if base_symbol == "RELIANCE" else base_symbol]
            mode = f"Stock: {base_symbol}"
            fallback_keywords = ['NSE', 'BSE', 'Sensex', 'Nifty']
        else:
            relevant_keywords = ['NSE', 'BSE', 'Indian stock market', 'Sensex', 'Nifty']
            mode = "Market"
            fallback_keywords = None
        
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
        
        if not news_items and stock_symbol and fallback_keywords:
            print(f"No {mode}-specific news found. Falling back to market news.")
            for entry in feed['entries']:
                title = entry.get('title', '').lower()
                description = entry.get('description', '').lower()
                text = title + ' ' + description
                if any(keyword.lower() in text for keyword in fallback_keywords):
                    news_items.append({
                        'title': entry.get('title', ''),
                        'description': entry.get('description', ''),
                        'pub_date': entry.get('published', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                        'note': 'Fallback: General market news'
                    })
        print(f"Fetched {len(news_items)} news items for {mode}")
        return news_items
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# Analyze sentiment using VADER
def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    return scores['compound']

# Fetch price data using yfinance
def fetch_price_data(stock_symbol=None):
    try:
        if stock_symbol:
            ticker = f"{stock_symbol}.NS"
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            return hist['Close'].iloc[-1] if not hist.empty else None, ticker
        else:
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="1d")
            return hist['Close'].iloc[-1] if not hist.empty else None, "^NSEI"
    except Exception as e:
        print(f"Error fetching price data: {e}")
        return None, None

# Process and display data
def process_data(stock_symbol=None):
    mode = f"Stock: {stock_symbol}" if stock_symbol else "Market (NSE/BSE)"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\nAnalyzing {mode} at {timestamp}")
    
    news_items = fetch_et_markets_news(stock_symbol)
    current_price, ticker = fetch_price_data(stock_symbol)
    
    if not news_items:
        print(f"No news found for {mode}.")
        return None, None
    
    data = []
    for item in news_items:
        text = item['title'] + ' ' + item['description']
        sentiment = analyze_sentiment(text)
        data.append({'title': item['title'], 'sentiment': sentiment, 'pub_date': item['pub_date'], 'note': item.get('note', '')})
    
    df = pd.DataFrame(data)
    avg_sentiment = df['sentiment'].mean()
    label = 'Bullish' if avg_sentiment > 0.05 else 'Bearish' if avg_sentiment < -0.05 else 'Neutral'
    
    # Display results in terminal
    print(f"\nPrice ({ticker}): {current_price:.2f}" if current_price else f"Price ({ticker}): N/A")
    print("\nSentiment Analysis Results:")
    for index, row in df.iterrows():
        print(f"- {row['title']} (Sentiment: {row['sentiment']:.2f}) {row['note']}")
    print(f"\nAverage Sentiment: {avg_sentiment:.2f} ({label})")
    
    # Plot with matplotlib
    plt.figure(figsize=(10, 6))
    colors = ['green' if s > 0 else 'red' if s < 0 else 'grey' for s in df['sentiment']]
    plt.bar(df['pub_date'], df['sentiment'], color=colors)
    plt.title(f"{mode} Sentiment Analysis")
    plt.xlabel("Publication Date")
    plt.ylabel("Sentiment Polarity (-1 to 1)")
    plt.ylim(-1, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    plot_file = f"sentiment_{stock_symbol if stock_symbol else 'market'}.png"
    plt.savefig(plot_file)
    print(f"Chart saved as '{plot_file}'")
    plt.show()
    
    # Save results to text file
    result_file = f"sentiment_results_{stock_symbol if stock_symbol else 'market'}.txt"
    with open(result_file, 'w') as f:
        f.write(f"{mode} Sentiment Analysis - {timestamp}\n")
        f.write(f"Price ({ticker}): {current_price:.2f if current_price else 'N/A'}\n\n")
        f.write("Sentiment Analysis Results:\n")
        for index, row in df.iterrows():
            f.write(f"- {row['title']} (Sentiment: {row['sentiment']:.2f}) {row['note']}\n")
        f.write(f"\nAverage Sentiment: {avg_sentiment:.2f} ({label})\n")
    print(f"Results saved to '{result_file}'")
    
    return df, current_price

# Main loop
def main():
    print("Stock Sentiment Analysis Tool")
    print("Enter an NSE stock symbol (e.g., 'RELIANCE') or leave blank for market sentiment.")
    print("Type 'exit' to quit.")
    
    while True:
        stock_symbol = input("\nStock symbol: ").strip()
        if stock_symbol.lower() == 'exit':
            print("Exiting...")
            break
        if not stock_symbol:
            stock_symbol = None
        process_data(stock_symbol)

if __name__ == "__main__":
    main()
