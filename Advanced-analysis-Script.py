import tweepy
from textblob import TextBlob
import configparser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter

# Read Twitter API credentials from config file
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

def get_tweets(query, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode="extended").items(count)
    return [tweet.full_text for tweet in tweets]

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analyze_sentiment(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def get_sentiment_score(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    return analysis.sentiment.polarity

def plot_sentiment_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment', data=df, palette='viridis')
    plt.title('Sentiment Distribution')
    plt.savefig('sentiment_distribution.png')
    plt.close()

def plot_sentiment_over_time(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df['sentiment_score'] = df['sentiment_score'].rolling(window=10).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['sentiment_score'])
    plt.title('Sentiment Score Over Time')
    plt.xlabel('Time')
    plt.ylabel('Sentiment Score (Moving Average)')
    plt.savefig('sentiment_over_time.png')
    plt.close()

def generate_wordcloud(df):
    all_words = ' '.join([tweet for tweet in df['clean_tweet']])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig('wordcloud.png')
    plt.close()

def get_top_hashtags(df):
    hashtags = []
    for tweet in df['tweet']:
        hashtags.extend([tag.lower() for tag in re.findall(r"#(\w+)", tweet)])
    return Counter(hashtags).most_common(10)

def main():
    search_term = input("Enter the search term for Twitter sentiment analysis: ")
    tweet_count = int(input("Enter the number of tweets to analyze: "))

    tweets = get_tweets(search_term, tweet_count)
    results = []
    for tweet in tweets:
        clean_tw = clean_tweet(tweet)
        sentiment = analyze_sentiment(clean_tw)
        sentiment_score = get_sentiment_score(clean_tw)
        results.append({
            'tweet': tweet,
            'clean_tweet': clean_tw,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'datetime': pd.Timestamp.now()  # Use actual tweet timestamp in production
        })

    df = pd.DataFrame(results)
    
    print("\nSentiment Distribution:")
    print(df['sentiment'].value_counts())
    
    print("\nTop 10 Hashtags:")
    print(get_top_hashtags(df))
    
    print("\nGenerating visualizations...")
    plot_sentiment_distribution(df)
    plot_sentiment_over_time(df)
    generate_wordcloud(df)
    
    print("\nSample tweets and their sentiment:")
    print(df[['tweet', 'sentiment', 'sentiment_score']].sample(5))
    
    # Export results to CSV
    df.to_csv(f'{search_term}_sentiment_analysis.csv', index=False)
    print(f"\nFull results exported to {search_term}_sentiment_analysis.csv")

if __name__ == "__main__":
    main()
