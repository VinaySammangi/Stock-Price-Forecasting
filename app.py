"""
Created on Wed Apr 20 14:21:59 2022

@author: vinaysammangi
"""

from flask import *
import json
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import random
import string
import pickle
import tweepy
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


app = Flask(__name__)

ticker_pattern = re.compile(r'(^\$[A-Z]+|^\$ES_F)')
ht_pattern = re.compile(r'#\w+')

charonly = re.compile(r'[^a-zA-Z\s]')
handle_pattern = re.compile(r'@\w+')
emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
url_pattern = re.compile(
    'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
pic_pattern = re.compile('pic\.twitter\.com/.{10}')
special_code = re.compile(r'(&amp;|&gt;|&lt;)')
tag_pattern = re.compile(r'<.*?>')

STOPWORDS = set(stopwords.words('english')).union(
    {'rt', 'retweet', 'RT', 'Retweet', 'RETWEET'})

lemmatizer = WordNetLemmatizer()

def hashtag(phrase):
    return ht_pattern.sub(' ', phrase)

def remove_ticker(phrase):
    return ticker_pattern.sub('', phrase)
    
def specialcode(phrase):
    return special_code.sub(' ', phrase)

def emoji(phrase):
    return emoji_pattern.sub(' ', phrase)

def url(phrase):
    return url_pattern.sub('', phrase)

def pic(phrase):
    return pic_pattern.sub('', phrase)

def html_tag(phrase):
    return tag_pattern.sub(' ', phrase)

def handle(phrase):
    return handle_pattern.sub('', phrase)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    
    # DIS, ticker symbol of Disney, is interpreted as the plural of "DI" 
    # in WordCloud, so I converted it to Disney
    phrase = re.sub('DIS', 'Disney', phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"(he|He)\'s", "he is", phrase)
    phrase = re.sub(r"(she|She)\'s", "she is", phrase)
    phrase = re.sub(r"(it|It)\'s", "it is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"(\'ve|has)", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def onlychar(phrase):
    return charonly.sub('', phrase)

def remove_stopwords(phrase):
    return " ".join([word for word in str(phrase).split()\
                     if word not in STOPWORDS])

def tokenize_stem(phrase):   
    tokens = word_tokenize(phrase)
    stem_words =[]
    for token in tokens:
        word = lemmatizer.lemmatize(token)
        stem_words.append(word)        
    buf = ' '.join(stem_words)    
    return buf

def arrange_text(ds):
    ds['text'] = ds['text'].str.strip().str.lower()
    ds['text'] = ds['text'].apply(emoji)
    ds['text'] = ds['text'].apply(handle)
    ds['text'] = ds['text'].apply(specialcode)
    ds['text'] = ds['text'].apply(hashtag)
    ds['text'] = ds['text'].apply(url)
    ds['text'] = ds['text'].apply(pic)
    ds['text'] = ds['text'].apply(html_tag)
    ds['text'] = ds['text'].apply(onlychar)
    ds['text'] = ds['text'].apply(decontracted)
    ds['text'] = ds['text'].apply(onlychar)
    ds['text'] = ds['text'].apply(tokenize_stem)
    ds['text'] = ds['text'].apply(remove_stopwords)
    return ds

def get_tweet_sentiments(ds):
    vec = pickle.load(open("vectorizer.pkl", 'rb'))
    nb_model = pickle.load(open("text_classifier.pkl", 'rb'))
    sentiments = []
    for text in ds["text"]:
        sentiments.append(nb_model.predict(vec.transform([text]))[0])
    ds["Sentiment"] = sentiments
    return ds

@app.route('/',methods =['GET'])
def home_page():
    data_set = {'Page':'Home','Message':'This is useless','Time':time.time()}
    json_dump = json.dumps(data_set)
    return json_dump

@app.route('/technical_forecast/',methods =['GET'])
def technical_forecast():
    tickerSymbol = str(request.args.get('ticker'))  #/technical_forecast/?ticker=AAPL
    tickerData = yf.Ticker(tickerSymbol) # Get ticker data
    end_date = pd.to_datetime('today')
    tickerDf_30m = tickerData.history(interval='30m', end=end_date).reset_index(drop=False) #get the historical prices for this ticker
    tickerDf_30m["Datetime"] = pd.to_datetime(pd.to_datetime(tickerDf_30m["Datetime"]).dt.strftime('%Y-%m-%d %H:%M:%S'))
    tickerDf_30m.drop(tickerDf_30m.tail(1).index,inplace=True)    
    tickerDf_30m =  tickerDf_30m[["Datetime","Open","High","Low","Close","Volume"]]
    tickerDf_30m['Datetime'] = tickerDf_30m['Datetime'].astype(str)
    forecast = tickerDf_30m.tail(7).Close.mean()
    data_set = {'Ticker':tickerSymbol,'DataFrame': tickerDf_30m.to_dict('list'),'Forecast':forecast,'ForecastConfidence':0.7}
    json_dump = json.dumps(data_set)
    return data_set

@app.route('/sentiment_forecast/',methods =['GET'])
def sentiment_forecast():
    tickerSymbol = str(request.args.get('ticker'))  #/sentiment_forecast/?ticker=AAPL
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAP1UbAEAAAAA3YuZnQ0qYxakT6wBZ42tzdjHs%2BQ%3DAtbLj2g6em2bZx4dqIEIQyW7sf1ttpfBErkmFOMc4h0DGIVp4e"

    client = tweepy.Client(bearer_token)
    response = client.search_all_tweets("#"+tickerSymbol+" stock", max_results=20,tweet_fields=['lang', 'created_at'])

    company_tweets = []
    for tweet in response.data:
        if tweet.lang=="en":
            company_tweets.append(tweet.text)
    
    company_tweets = list(set(company_tweets))
    tweets_data = pd.DataFrame(company_tweets)
    tweets_data.columns = ["text"]
    tweets_data = arrange_text(tweets_data)
    tweets_data.drop_duplicates(inplace=True)
    tweets_data = get_tweet_sentiments(tweets_data)
    tweets_data.columns = ["Tweet","Sentiment"]
    tweets_data = tweets_data.loc[tweets_data["Tweet"]!="",]
    data_set = {'Ticker':tickerSymbol,'DataFrame': tweets_data.to_dict('list'),'Forecast':'Bullish','ForecastConfidence':0.8}
    json_dump = json.dumps(data_set)   
    return data_set

if __name__ == '__main__':
    app.run(port=7777)