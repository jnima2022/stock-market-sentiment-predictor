# Stock Market Sentiment Analysis

This project analyzes the impact of news sentiment on stock prices for major tech companies using Python.

## Project Description

This analysis explores the relationship between news sentiment and stock price movements for tech giants like MSFT, AMZN, TSLA, GOOGL, NVDA, AAPL, and META. It involves sentiment analysis of news articles, statistical analysis of correlations, and machine learning predictions.

## My Goal

The goal of this project is to understand how news sentiment affects stock prices and to explore the potential of using sentiment data for predicting stock market trends.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- TextBlob

## Installation

1. Clone this repository
2. Navigate to the project directory
3. Install required packages

## Data Sources

- Stock price data: [ALPHA_VANTAGE_API]
- News articles: [NEWSAPI_KEY]

## Main Scripts

- `data_collection.py`: Collects stock price and news data using APIs
- `sentiment_analysis.py`: Performs sentiment analysis on news articles
- `statistical_analysis.py`: Conducts correlation analysis and visualizations
- `data_visualization.py`: Creates visualizations of the data
- `advanced_analysis.py`: Conducts advanced statistical tests and analyses
- `advanced_ml_analysis.py`: Implements machine learning models for stock price prediction

## Usage

Run the scripts in the following order:

1. `python data_collection.py`
2. `python sentiment_analysis.py`
3. `python statistical_analysis.py`
4. `python data_visualization.py`
5. `python advanced_analysis.py`
6. `python advanced_ml_analysis.py`

Check the generated PNG files and console output for results and visualizations.

## Key Findings and Insights

- Weak correlations between news sentiment and stock prices were observed.
- Most correlations were not statistically significant.
- Sentiment scores showed limited predictive power for stock price movements.

## Machine Learning

- **Model Used**: Random Forest Regression
- **Performance**: Achieved an R-squared score of 0.998, indicating strong predictive power.

## Future Improvements

- Extend analysis to longer periods for more comprehensive insights.
- Incorporate advanced NLP techniques for improved sentiment analysis.
- Explore non-linear relationships between sentiment and stock prices.
