import pandas as pd
from textblob import TextBlob
import re
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

def categorize_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_sentiment(df):
    # Clean the text
    df['cleaned_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['cleaned_text'] = df['cleaned_text'].apply(clean_text)
    
    # Get sentiment scores
    df['sentiment_score'] = df['cleaned_text'].apply(get_sentiment_score)
    
    # Categorize sentiment
    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
    
    return df

def main():
    # Load the news data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    news_data_path = os.path.join(data_dir, 'stock_news_data.csv')
    df = pd.read_csv(news_data_path)
    
    # Perform sentiment analysis
    df_with_sentiment = analyze_sentiment(df)
    
    # Save the results
    output_path = os.path.join(data_dir, 'stock_news_with_sentiment.csv')
    df_with_sentiment.to_csv(output_path, index=False)
    print(f"Sentiment analysis completed. Results saved to {output_path}")
    
    # Print some statistics
    print("\nSentiment Distribution:")
    print(df_with_sentiment['sentiment_category'].value_counts(normalize=True))
    
    print("\nAverage Sentiment Score by Ticker:")
    print(df_with_sentiment.groupby('ticker')['sentiment_score'].mean())

if __name__ == "__main__":
    main()