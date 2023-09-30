import tweepy
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set your Twitter API credentials
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Authenticate to Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Define a search query and the number of tweets to fetch
search_query = 'your_search_query'
tweet_count = 100

# Fetch tweets
tweets = api.search(q=search_query, count=tweet_count)

# Function to perform sentiment analysis using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Analyze sentiment for each tweet
sentiment_results = [(tweet.text, get_sentiment(tweet.text)) for tweet in tweets]

# Create a Pandas DataFrame for analysis
df = pd.DataFrame(sentiment_results, columns=["Tweet", "Sentiment"])

# Plot sentiment distribution
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.countplot(x="Sentiment", data=df, palette="Set3")
plt.title("Sentiment Analysis")
plt.show()
