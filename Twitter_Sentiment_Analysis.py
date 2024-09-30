# -*- coding: utf-8 -*-
"""
Advanced Twitter Analysis Script
"""
from tweepy import API, OAuthHandler, Cursor
from io import BytesIO
import base64
from textblob import TextBlob
import Twitter_credentials
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import asyncio
import aiohttp
from collections import Counter
import emoji

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TwitterAuthenticator():
    def authenticate_twitter_app(self):
        auth = OAuthHandler(Twitter_credentials.CONSUMER_KEY, Twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(Twitter_credentials.ACCESS_TOKEN, Twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth

class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

class TweetAnalyzer():
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_tweet(self, tweet):
        # Remove URLs, RT, @mentions
        tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)
        tweet = re.sub(r'\brt\b', '', tweet.lower())
        # Remove punctuations and numbers
        tweet = re.sub("[^a-zA-Z]", " ", tweet)
        # Remove stopwords
        words = word_tokenize(tweet)
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(words)

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def tweets_to_data_frame(self, tweets):  
        df = pd.DataFrame([tweet.full_text for tweet in tweets], columns=['tweets'])
        df['id'] = np.array([tweet.id for tweet in tweets])
        df['len'] = np.array([len(tweet.full_text) for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
        df['sentiment'] = np.array([self.analyze_sentiment(tweet.full_text) for tweet in tweets])
        df['polarity'] = np.array([self.Polarity(tweet.full_text) for tweet in tweets])
        df['subjectivity'] = np.array([self.subjectivity(tweet.full_text) for tweet in tweets])
        df['clean_tweet'] = np.array([self.clean_tweet(tweet.full_text) for tweet in tweets])
        return df

    def Polarity(self, tweet):
        return TextBlob(self.clean_tweet(tweet)).sentiment.polarity
    
    def subjectivity(self, tweet):
        return TextBlob(self.clean_tweet(tweet)).sentiment.subjectivity

    def get_topic_model(self, df, num_topics=5, num_words=10):
        vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(df['clean_tweet'].values.astype('U'))
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(doc_term_matrix)
        topics = []
        for idx, topic in enumerate(lda.components_):
            topics.append([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-num_words - 1:-1]])
        return topics

    def get_hashtags(self, tweets):
        hashtags = []
        for tweet in tweets:
            hashtags.extend([tag['text'] for tag in tweet.entities.get('hashtags', [])])
        return Counter(hashtags)

    def get_mentions(self, tweets):
        mentions = []
        for tweet in tweets:
            mentions.extend([mention['screen_name'] for mention in tweet.entities.get('user_mentions', [])])
        return Counter(mentions)

    def get_emoji_distribution(self, tweets):
        emoji_list = []
        for tweet in tweets:
            emoji_list.extend([c for c in tweet.full_text if c in emoji.UNICODE_EMOJI['en']])
        return Counter(emoji_list)

class keyword():
    @staticmethod
    def key(word, count=200):
        twitter_client = TwitterClient()
        tweet_analyzer = TweetAnalyzer()
        api = twitter_client.get_twitter_client_api()
        tweets = api.user_timeline(screen_name=word, count=count, tweet_mode='extended')
        df = tweet_analyzer.tweets_to_data_frame(tweets)
        return df, tweets

class plotting():
    @staticmethod
    def show_wordcloud(data, title=None):
        img = BytesIO()
        stopwords = set(STOPWORDS)
        wd = WordCloud(
            background_color='white',
            stopwords=stopwords,
            max_words=200,
            max_font_size=40, 
            scale=3,
            random_state=1
        ).generate(str(data))
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wd, interpolation='bilinear')
        plt.axis('off')
        if title:
            plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode('utf-8')
    
    @staticmethod
    def sentiment(df):
        plt.figure(figsize=(10, 5))
        sns.countplot(x="sentiment", data=df, palette="Blues_d")
        plt.title("Sentiment Distribution")
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode('utf-8')

    @staticmethod
    def PolarityAndSubjectivity(df):
        plt.figure(figsize=(10, 8))
        plt.scatter(df['polarity'], df['subjectivity'], alpha=0.5)
        plt.title('Sentiment Analysis', fontsize=20)
        plt.xlabel('Polarity', fontsize=15)
        plt.ylabel('Subjectivity', fontsize=15)
        plt.tight_layout()
        img = BytesIO()
        plt.savefig(img, format='png', dpi=300)
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode('utf-8')

    @staticmethod
    def plot_time_series(df):
        plt.figure(figsize=(12, 6))
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df['sentiment'].resample('D').mean().plot(kind='line')
        plt.title('Sentiment Over Time')
        plt.ylabel('Average Sentiment')
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode('utf-8')

    @staticmethod
    def plot_hashtag_distribution(hashtags, top_n=10):
        plt.figure(figsize=(10, 5))
        hashtags.most_common(top_n)[::-1].plot.barh()
        plt.title(f'Top {top_n} Hashtags')
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode('utf-8')

async def fetch_user_info(session, user_id):
    url = f"https://api.twitter.com/2/users/{user_id}"
    headers = {"Authorization": f"Bearer {Twitter_credentials.BEARER_TOKEN}"}
    async with session.get(url, headers=headers) as response:
        return await response.json()

async def get_user_info(user_ids):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_user_info(session, user_id) for user_id in user_ids]
        return await asyncio.gather(*tasks)

if __name__ == '__main__':
    # Example usage
    username = "elonmusk"
    df, tweets = keyword.key(username, count=1000)
    
    # Basic analysis
    print(f"Total tweets analyzed: {len(df)}")
    print(f"Average sentiment: {df['sentiment'].mean()}")
    
    # Generate and save plots
    wordcloud = plotting.show_wordcloud(df['clean_tweet'], title=f"Word Cloud for @{username}")
    sentiment_plot = plotting.sentiment(df)
    polarity_subjectivity_plot = plotting.PolarityAndSubjectivity(df)
    time_series_plot = plotting.time_series(df)
    
    # Topic modeling
    tweet_analyzer = TweetAnalyzer()
    topics = tweet_analyzer.get_topic_model(df)
    print("Top topics:")
    for idx, topic in enumerate(topics):
        print(f"Topic {idx + 1}: {', '.join(topic)}")
    
    # Hashtag analysis
    hashtags = tweet_analyzer.get_hashtags(tweets)
    hashtag_plot = plotting.plot_hashtag_distribution(hashtags)
    
    # Mention analysis
    mentions = tweet_analyzer.get_mentions(tweets)
    print("Top mentions:", mentions.most_common(5))
    
    # Emoji analysis
    emojis = tweet_analyzer.get_emoji_distribution(tweets)
    print("Top emojis:", emojis.most_common(5))
    
    # Asynchronous user info fetching (for mentioned users)
    top_mentioned_users = [mention[0] for mention in mentions.most_common(5)]
    user_info = asyncio.run(get_user_info(top_mentioned_users))
    print("Info for top mentioned users:", user_info)
