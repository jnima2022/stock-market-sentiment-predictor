import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API keys from environment variables
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')


def fetch_news(ticker, days=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://newsapi.org/v2/everything?q={ticker}&from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&sortBy=popularity&apiKey={NEWSAPI_KEY}"
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['articles']
    else:
        print(f"Error fetching news: {response.status_code}")
        print(f"Response content: {response.text}")
        return None

def fetch_stock_data(ticker):
    function = 'TIME_SERIES_DAILY'
    url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}'
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'Time Series (Daily)' in data:
            df = pd.DataFrame(data['Time Series (Daily)']).T
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
            df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': int})
            df.reset_index(inplace=True)  # Reset index to make date a column
            df.rename(columns={'index': 'date'}, inplace=True)  # Rename 'index' to 'date'
            df['Ticker'] = ticker  # Add ticker column
            return df
        else:
            print(f"Error: 'Time Series (Daily)' not found in response for {ticker}")
            print(f"Response content: {data}")
    else:
        print(f"Error fetching stock data for {ticker}: Status code {response.status_code}")
        print(f"Response content: {response.text}")
    return None

def is_relevant_article(title, content):
    relevant_keywords = ['stock', 'market', 'finance', 'investor', 'trading', 'economy']
    title_text = title.lower() if title else ''
    content_text = content.lower() if content else ''
    return any(keyword in title_text or keyword in content_text for keyword in relevant_keywords)

def process_news(ticker):
    news_data = fetch_news(ticker)
    if news_data:
        df = pd.DataFrame(news_data)
        df['ticker'] = ticker
        df['date'] = pd.to_datetime(df['publishedAt'])
        
        # Apply the relevance filter
        df['is_relevant'] = df.apply(lambda row: is_relevant_article(row['title'], row['description']), axis=1)
        df = df[df['is_relevant']]  # Keep only relevant articles
        
        return df[['ticker', 'date', 'title', 'description', 'content', 'source', 'url']]
    return pd.DataFrame()  # Return an empty DataFrame if no news data

def collect_data(tickers):
    all_news = pd.DataFrame()
    all_stock_data = pd.DataFrame()

    for ticker in tickers:
        print(f"Processing {ticker}...")
        
        try:
            ticker_news = process_news(ticker)
            all_news = pd.concat([all_news, ticker_news], ignore_index=True)
            
            stock_data = fetch_stock_data(ticker)
            if stock_data is not None:
                stock_data['Ticker'] = ticker
                all_stock_data = pd.concat([all_stock_data, stock_data], ignore_index=True)
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
        
        # Add a delay to avoid hitting API rate limits
        time.sleep(15)  # 15 seconds delay between each ticker

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    all_news.to_csv(os.path.join(data_dir, 'stock_news_data.csv'), index=False)
    print("News data saved to stock_news_data.csv")

    all_stock_data.to_csv(os.path.join(data_dir, 'stock_price_data.csv'), index=False)
    print("Stock price data saved to stock_price_data.csv")

if __name__ == "__main__":
    tickers = ['NVDA','TSLA','AAPL','MSFT','META','GOOGL','AMZN']  # Example tickers
    collect_data(tickers)
