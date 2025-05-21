import requests
import os
from datetime import datetime, timedelta
import time
import re
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize NLTK
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# News API key
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/everything"

def get_news(crypto_symbol):
    """
    Fetch news articles related to a cryptocurrency
    
    Parameters:
    crypto_symbol (str): Cryptocurrency symbol, e.g., 'BTC'
    
    Returns:
    list: List of news articles with title, source, published date, and content
    """
    # Map of crypto symbols to full names for better news search
    crypto_names = {
        "BTC": "Bitcoin",
        "ETH": "Ethereum",
        "BNB": "Binance Coin",
        "XRP": "Ripple",
        "ADA": "Cardano",
        "SOL": "Solana",
        "DOT": "Polkadot",
        "DOGE": "Dogecoin",
        "AVAX": "Avalanche",
        "SHIB": "Shiba Inu"
    }
    
    # If no API key is provided, return some sample data with neutral sentiment
    if not NEWS_API_KEY:
        # Return sample news with neutral sentiment
        return get_sample_news(crypto_symbol)
    
    # Get the full name if available, otherwise use the symbol
    query_term = crypto_names.get(crypto_symbol, crypto_symbol)
    
    # Calculate date range (7 days ago until now)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    params = {
        'q': query_term + " cryptocurrency",
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'language': 'en',
        'sortBy': 'publishedAt',
        'apiKey': NEWS_API_KEY
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()
        
        news_data = response.json()
        articles = news_data.get('articles', [])
        
        # Process and format articles
        processed_articles = []
        for article in articles[:15]:  # Limit to top 15 articles
            processed_articles.append({
                'title': article.get('title', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'published_at': format_date(article.get('publishedAt', '')),
                'url': article.get('url', ''),
                'summary': article.get('description', ''),
                'content': clean_content(article.get('content', ''))
            })
        
        return processed_articles
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return get_sample_news(crypto_symbol)

def get_sample_news(crypto_symbol):
    """
    Return sample neutral news for when API key is not available
    """
    return [
        {
            'title': f"{crypto_symbol} Price Analysis: Market Shows Consolidation",
            'source': "Crypto News",
            'published_at': datetime.now().strftime('%Y-%m-%d'),
            'url': "#",
            'summary': f"{crypto_symbol} markets have been relatively stable in recent days as traders await further signals.",
            'content': f"The {crypto_symbol} market has been consolidating in a range as volume decreases. Analysts are divided on the next move."
        }
    ]

def format_date(date_str):
    """Format the date string for better display"""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
        return date_obj.strftime('%Y-%m-%d %H:%M')
    except (ValueError, TypeError):
        return date_str

def clean_content(content):
    """Clean article content by removing character count markers"""
    if not content:
        return ""
    # Remove patterns like "[+1234 chars]" that often appear in NewsAPI content
    return re.sub(r'\[\+\d+ chars\]', '', content)

def analyze_sentiment(news_data):
    """
    Analyze sentiment of news articles
    
    Parameters:
    news_data (list): List of news articles
    
    Returns:
    float: Sentiment score between -1 (very negative) and 1 (very positive)
    """
    if not news_data:
        return 0.0  # Neutral sentiment if no data
    
    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Analyze each article and collect scores
    scores = []
    
    for article in news_data:
        # Combine title and content for analysis
        text = article['title'] + ' ' + article['content']
        
        # Get sentiment scores
        vs = analyzer.polarity_scores(text)
        
        # We'll use the compound score which is a normalized score between -1 and 1
        scores.append(vs['compound'])
    
    # Average the scores if we have any
    if scores:
        # We give more weight to more recent news
        weights = [1.0 - (idx * 0.05) for idx in range(len(scores))]
        weights = [max(0.5, w) for w in weights]  # Ensure minimum weight of 0.5
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate weighted average
        weighted_score = sum(s * w for s, w in zip(scores, weights))
        return weighted_score
    
    return 0.0  # Neutral sentiment if no scores
