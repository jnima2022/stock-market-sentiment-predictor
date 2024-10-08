import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    news_sentiment_path = os.path.join(data_dir, 'stock_news_with_sentiment.csv')
    stock_price_path = os.path.join(data_dir, 'stock_price_data.csv')
    
    news_sentiment_df = pd.read_csv(news_sentiment_path)
    stock_price_df = pd.read_csv(stock_price_path)
    
    print("News Sentiment DataFrame columns:", news_sentiment_df.columns)
    print("Stock Price DataFrame columns:", stock_price_df.columns)
    
    print("News Sentiment DataFrame sample:")
    print(news_sentiment_df.head())
    print("\nStock Price DataFrame sample:")
    print(stock_price_df.head())
    
    # Convert date columns to datetime and remove timezone information
    news_sentiment_df['date'] = pd.to_datetime(news_sentiment_df['date']).dt.tz_localize(None)
    stock_price_df['date'] = pd.to_datetime(stock_price_df['date'])
    
    print("\nDate column types after conversion:")
    print("News Sentiment date type:", news_sentiment_df['date'].dtype)
    print("Stock Price date type:", stock_price_df['date'].dtype)
    
    return news_sentiment_df, stock_price_df

def plot_sentiment_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment_category', data=df)
    plt.title('Distribution of Sentiment Categories')
    plt.savefig('sentiment_distribution.png')
    plt.close()

def plot_average_sentiment_by_ticker(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='ticker', y='sentiment_score', data=df)
    plt.title('Average Sentiment Score by Ticker')
    plt.savefig('average_sentiment_by_ticker.png')
    plt.close()

def plot_sentiment_over_time(df):
    plt.figure(figsize=(14, 8))
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker]
        plt.plot(ticker_df['date'], ticker_df['sentiment_score'], label=ticker)
    plt.title('Sentiment Score Over Time by Ticker')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.legend()
    plt.savefig('sentiment_over_time.png')
    plt.close()

def plot_sentiment_vs_price(news_df, price_df):
    plt.figure(figsize=(14, 8))
    for ticker in news_df['ticker'].unique():
        ticker_news = news_df[news_df['ticker'] == ticker]
        ticker_price = price_df[price_df['Ticker'] == ticker]
        
        if ticker_price.empty:
            print(f"No price data found for ticker {ticker}")
            continue
        
        # Merge news sentiment with stock price data
        merged_data = pd.merge_asof(ticker_news.sort_values('date'), 
                                    ticker_price.sort_values('date'), 
                                    left_on='date', 
                                    right_on='date', 
                                    direction='nearest',
                                    tolerance=pd.Timedelta('1d'))
        
        if 'Close' in merged_data.columns and not merged_data.empty:
            plt.scatter(merged_data['sentiment_score'], merged_data['Close'], label=ticker, alpha=0.5)
        else:
            print(f"Warning: 'Close' column not found or no data for ticker {ticker}")
    
    plt.title('Sentiment Score vs. Stock Price')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Stock Price (Close)')
    plt.legend()
    plt.savefig('sentiment_vs_price.png')
    plt.close()
    
def main():
    news_sentiment_df, stock_price_df = load_data()
    
    plot_sentiment_distribution(news_sentiment_df)
    plot_average_sentiment_by_ticker(news_sentiment_df)
    plot_sentiment_over_time(news_sentiment_df)
    plot_sentiment_vs_price(news_sentiment_df, stock_price_df)
    
    print("Visualizations have been saved as PNG files in the current directory.")

if __name__ == "__main__":
    main()