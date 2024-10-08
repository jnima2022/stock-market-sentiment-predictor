import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    merged_data_path = os.path.join(data_dir, 'merged_data.csv')
    
    print(f"Attempting to load data from: {merged_data_path}")
    
    if os.path.exists(merged_data_path):
        data = pd.read_csv(merged_data_path)
        data['date'] = pd.to_datetime(data['date'])
        print(f"Successfully loaded data with shape: {data.shape}")
        return data
    else:
        print(f"Error: File not found at {merged_data_path}")
        print("Current working directory:", os.getcwd())
        print("Contents of the data directory:")
        print(os.listdir(data_dir))
        raise FileNotFoundError(f"No such file or directory: '{merged_data_path}'")

def prepare_features(data):
    # Create lagged features and technical indicators
    for lag in [1, 3, 7]:
        data[f'sentiment_lag_{lag}'] = data.groupby('ticker')['sentiment_score'].shift(lag)
    
    # Add simple moving averages
    for window in [5, 10, 20]:
        data[f'price_sma_{window}'] = data.groupby('ticker')['Close'].rolling(window=window).mean().reset_index(0, drop=True)
    
    # Add relative strength index (RSI)
    delta = data.groupby('ticker')['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data.dropna()

def train_random_forest(data, target='Close'):
    features = ['sentiment_score', 'sentiment_lag_1', 'sentiment_lag_3', 'sentiment_lag_7', 
                'price_sma_5', 'price_sma_10', 'price_sma_20', 'RSI']
    
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    
    # Feature importance
    feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model, X_test, y_test, feature_importance

def perform_granger_causality(data):
    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker].set_index('date')
        variables = ['sentiment_score', 'Close']
        max_lag = 1  # Reduce this to 1
        
        print(f"\nGranger Causality Test for {ticker}:")
        try:
            grangercausalitytests(ticker_data[variables], maxlag=max_lag, verbose=True)
        except ValueError as e:
            print(f"Error performing Granger Causality Test for {ticker}: {str(e)}")

def plot_feature_importance(feature_importance):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance in Random Forest Model')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_predicted_vs_actual(model, X_test, y_test):
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Predicted vs Actual Stock Prices')
    plt.tight_layout()
    plt.savefig('predicted_vs_actual.png')
    plt.close()

def analyze_sentiment_impact(data):
    plt.figure(figsize=(12, 6))
    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker]
        plt.scatter(ticker_data['sentiment_score'], ticker_data['Close'], alpha=0.5, label=ticker)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Stock Price (Close)')
    plt.title('Sentiment Score vs Stock Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sentiment_vs_price.png')
    plt.close()
    
def main():
    data = load_data()
    data = prepare_features(data)
    
    print("Training Random Forest Model:")
    model, X_test, y_test, feature_importance = train_random_forest(data)
    
    plot_feature_importance(feature_importance)
    plot_predicted_vs_actual(model, X_test, y_test)
    analyze_sentiment_impact(data)
    
    print("\nPerforming Granger Causality Tests:")
    perform_granger_causality(data)

    print("\nAdvanced analysis and visualizations completed. Check the generated PNG files.")

if __name__ == "__main__":
    main()