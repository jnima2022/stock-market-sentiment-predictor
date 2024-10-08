import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_merged_data():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    merged_data_path = os.path.join(data_dir, 'merged_data.csv')
    
    if os.path.exists(merged_data_path):
        data = pd.read_csv(merged_data_path)
        data['date'] = pd.to_datetime(data['date'])
        return data
    else:
        print("Merged data file not found. Please run the statistical_analysis.py script first.")
        sys.exit(1)

def calculate_moving_averages(data, windows=[3, 7]):
    for window in windows:
        data[f'sentiment_ma_{window}'] = data.groupby('ticker')['sentiment_score'].rolling(window=window).mean().reset_index(0, drop=True)
    return data

def aggregate_weekly_data(data):
    data['week'] = data['date'].dt.to_period('W')
    weekly_data = data.groupby(['ticker', 'week']).agg({
        'sentiment_score': 'mean',
        'Close': ['first', 'last']
    }).reset_index()
    weekly_data.columns = ['ticker', 'week', 'avg_sentiment', 'price_open', 'price_close']
    weekly_data['price_change'] = (weekly_data['price_close'] - weekly_data['price_open']) / weekly_data['price_open']
    return weekly_data

def calculate_spearman_correlation(data):
    correlations = {}
    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker].dropna(subset=['sentiment_score', 'Close'])
        if len(ticker_data) > 1:
            correlation, p_value = stats.spearmanr(ticker_data['sentiment_score'], ticker_data['Close'])
            correlations[ticker] = {'correlation': correlation, 'p_value': p_value}
        else:
            print(f"Not enough data for Spearman correlation for {ticker}")
    return correlations

def calculate_mutual_information(data):
    mi_scores = {}
    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker].dropna(subset=['sentiment_score', 'Close'])
        if len(ticker_data) > 1:
            mi = mutual_info_regression(ticker_data[['sentiment_score']], ticker_data['Close'])[0]
            mi_scores[ticker] = mi
        else:
            print(f"Not enough data for Mutual Information calculation for {ticker}")
    return mi_scores

def plot_weekly_sentiment_vs_price_change(weekly_data):
    plt.figure(figsize=(12, 8))
    for ticker in weekly_data['ticker'].unique():
        ticker_data = weekly_data[weekly_data['ticker'] == ticker]
        plt.scatter(ticker_data['avg_sentiment'], ticker_data['price_change'], label=ticker, alpha=0.7)
    
    plt.xlabel('Weekly Average Sentiment')
    plt.ylabel('Weekly Price Change (%)')
    plt.title('Weekly Sentiment vs Price Change')
    plt.legend()
    plt.savefig('weekly_sentiment_vs_price_change.png')
    plt.close()

def main():
    # Load the merged data
    merged_data = load_merged_data()

    # Print data info
    print("Data info:")
    print(merged_data.info())
    
    print("\nMissing values:")
    print(merged_data.isnull().sum())

    # Calculate moving averages
    merged_data = calculate_moving_averages(merged_data)

    # Aggregate weekly data
    weekly_data = aggregate_weekly_data(merged_data)

    # Calculate Spearman correlation
    spearman_correlations = calculate_spearman_correlation(merged_data)
    print("\nSpearman Correlations:")
    for ticker, corr_data in spearman_correlations.items():
        print(f"{ticker}: Correlation = {corr_data['correlation']:.4f}, P-value = {corr_data['p_value']:.4f}")

    # Calculate Mutual Information
    mi_scores = calculate_mutual_information(merged_data)
    print("\nMutual Information Scores:")
    for ticker, mi in mi_scores.items():
        print(f"{ticker}: MI = {mi:.4f}")

    # Plot weekly sentiment vs price change
    plot_weekly_sentiment_vs_price_change(weekly_data)

    print("\nAdvanced analysis completed. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()