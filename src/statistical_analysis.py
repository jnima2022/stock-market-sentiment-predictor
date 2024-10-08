import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
    
    news_sentiment_df['date'] = pd.to_datetime(news_sentiment_df['date']).dt.tz_localize(None)
    stock_price_df['date'] = pd.to_datetime(stock_price_df['date'])

    print("News sentiment columns:", news_sentiment_df.columns)
    print("Stock price columns:", stock_price_df.columns)
    
    return news_sentiment_df, stock_price_df

def calculate_correlations(merged_data):
    correlations = {}
    for ticker in merged_data['ticker'].unique():
        ticker_data = merged_data[merged_data['ticker'] == ticker]
        sentiment_data = ticker_data['sentiment_score'].dropna()
        price_data = ticker_data['Close'].dropna()
        
        print(f"\nAnalyzing {ticker}:")
        print(f"  Sentiment data points: {len(sentiment_data)}")
        print(f"  Price data points: {len(price_data)}")
        
        if len(sentiment_data) > 1 and len(price_data) > 1:
            # Align the data
            aligned_data = pd.concat([sentiment_data, price_data], axis=1).dropna()
            if len(aligned_data) > 1:
                correlation, p_value = stats.pearsonr(aligned_data['sentiment_score'], aligned_data['Close'])
                correlations[ticker] = {'correlation': correlation, 'p_value': p_value}
                print(f"  Correlation calculated with {len(aligned_data)} aligned data points")
            else:
                print(f"  Not enough aligned data points to calculate correlation for {ticker}")
        else:
            print(f"  Not enough data to calculate correlation for {ticker}")
    return correlations

def print_sample_data(merged_data):
    print("\nSample of merged data:")
    print(merged_data.sample(5).to_string())

def create_lagged_features(merged_data, lag_days=[1, 3, 7]):
    for lag in lag_days:
        merged_data[f'sentiment_score_lag_{lag}'] = merged_data.groupby('ticker')['sentiment_score'].shift(lag)
    return merged_data

def plot_correlation_heatmap(correlations):
    corr_data = pd.DataFrame(correlations).T
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_data[['correlation']], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation between Sentiment Score and Stock Price')
    plt.savefig('correlation_heatmap.png')
    plt.close()

def plot_lagged_correlations(merged_data, lag_days=[1, 3, 7]):
    plt.figure(figsize=(12, 6))
    for ticker in merged_data['ticker'].unique():
        ticker_data = merged_data[merged_data['ticker'] == ticker].sort_values('date')
        correlations = []
        for lag in lag_days:
            lagged_sentiment = ticker_data[f'sentiment_score_lag_{lag}']
            price = ticker_data['Close']
            # Align the data and drop NaN values
            aligned_data = pd.concat([lagged_sentiment, price], axis=1).dropna()
            if len(aligned_data) > 1:
                correlation, _ = stats.pearsonr(aligned_data[f'sentiment_score_lag_{lag}'], aligned_data['Close'])
                correlations.append(correlation)
            else:
                correlations.append(np.nan)
        plt.plot(lag_days, correlations, marker='o', label=ticker)
    
    plt.xlabel('Lag (days)')
    plt.ylabel('Correlation')
    plt.title('Lagged Correlations between Sentiment Score and Stock Price')
    plt.legend()
    plt.savefig('lagged_correlations.png')
    plt.close()

def plot_sentiment_vs_price_scatter(merged_data):
    for ticker in merged_data['ticker'].unique():
        ticker_data = merged_data[merged_data['ticker'] == ticker]
        plt.figure(figsize=(10, 6))
        plt.scatter(ticker_data['sentiment_score'], ticker_data['Close'])
        plt.title(f'Sentiment Score vs. Stock Price for {ticker}')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Stock Price (Close)')
        plt.savefig(f'sentiment_vs_price_scatter_{ticker}.png')
        plt.close()

def plot_sentiment_and_price_time_series(merged_data):
    for ticker in merged_data['ticker'].unique():
        ticker_data = merged_data[merged_data['ticker'] == ticker].sort_values('date')
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price (Close)', color='tab:blue')
        ax1.plot(ticker_data['date'], ticker_data['Close'], color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Sentiment Score', color='tab:orange')
        ax2.plot(ticker_data['date'], ticker_data['sentiment_score'], color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        plt.title(f'Sentiment Score and Stock Price Over Time for {ticker}')
        plt.savefig(f'sentiment_and_price_time_series_{ticker}.png')
        plt.close()

def check_missing_data(merged_data):
    print("\nMissing data summary:")
    print(merged_data.isnull().sum())
    
    print("\nNumber of rows per ticker:")
    print(merged_data['ticker'].value_counts())

def main():
    news_sentiment_df, stock_price_df = load_data()
    
    # Rename 'Ticker' to 'ticker' in stock_price_df for consistency
    stock_price_df = stock_price_df.rename(columns={'Ticker': 'ticker'})
    
    # Merge data
    merged_data = pd.merge_asof(news_sentiment_df.sort_values('date'), 
                                stock_price_df.sort_values('date'), 
                                left_on='date', 
                                right_on='date', 
                                by='ticker',
                                direction='nearest',
                                tolerance=pd.Timedelta('1d'))
    
    # Check if the merge was successful
    print("Merged data columns:", merged_data.columns)
    print("Merged data shape:", merged_data.shape)
    
    # Check for missing data
    check_missing_data(merged_data)
    
    # Print sample data
    print_sample_data(merged_data)
    
    # Calculate correlations
    correlations = calculate_correlations(merged_data)
    print("\nCorrelations between sentiment score and stock price:")
    for ticker, corr_data in correlations.items():
        print(f"{ticker}: Correlation = {corr_data['correlation']:.4f}, P-value = {corr_data['p_value']:.4f}")
    
    # Create lagged features
    merged_data = create_lagged_features(merged_data)
    
    # Plot correlation heatmap
    plot_correlation_heatmap(correlations)
    
    # Plot lagged correlations
    plot_lagged_correlations(merged_data)
    
    # Plot scatter plots
    plot_sentiment_vs_price_scatter(merged_data)

    # Plot time series
    plot_sentiment_and_price_time_series(merged_data)

    # Save the merged data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    merged_data_path = os.path.join(data_dir, 'merged_data.csv')
    merged_data.to_csv(merged_data_path, index=False)
    print(f"Merged data saved to {merged_data_path}")
    
    print("\nStatistical analysis and visualizations completed. Check the generated PNG files.")

if __name__ == "__main__":
    main()