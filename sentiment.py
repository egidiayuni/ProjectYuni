import tweepy
from textblob import TextBlob
import re
import pandas as pd

# Step 1 - Authentikasi API
consumer_key = 'gsQfbJhVojdnqrQ0JSOq0Jr8R'
consumer_secret = 'cJNvNEKo94yI095hPgg5dToNUplfnpu6L2R46C1pXQr2GHNyCy'

access_token = '1600315706-VXqTNvKsWkvmTNtlzi0Y35qN0sisJnTHMCb1yxI'
access_token_secret = 'Fuxo6nMR0PuJ6ELQJ6e9q9y2GAr7Je14RNIBQ89jrIbip'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Step 2 function untuk mengecek sentiment dari tweet.


def polarity(analysis):
    threshold = 0
    if analysis == threshold:
        return 'Neutral'
    elif analysis > threshold:
        return 'Positive'
    else:
        return 'Negative'

# Step 3 Untuk membersihkan tweets dengan menghilangkan special karakter yang tidak diperlukan dalam analisi


def clean_tweet(tweet):
    cleanedtweet = ' '.join(
        re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+'')", " ", tweet).split())
    return cleanedtweet


# Step 4 - Memasukan Tweets
lang = 'id'  # lang atur di sini
topic = input("Cari Tweet: ")

max_tweets = 1000 # banyaknya tweet yang mau diambil atur sini
searched_tweets = []

while len(searched_tweets) < max_tweets:
    remaining_tweets = max_tweets - len(searched_tweets)
    try:
        new_tweets = api.search(q=topic, lang=lang, count=remaining_tweets)
#                                    geocode=geocode)
        print('found', len(new_tweets), 'tweets')
        if not new_tweets:
            print('no tweets found')
            break
        searched_tweets.extend(new_tweets)
        max_id = new_tweets[-1].id
    except tweepy.TweepError:
        print('exception raised, waiting 15 minutes')
        time.sleep(15*60)
        break  # stop the loop

tweet_txt = []
tweet_stm = []

# Step 5 Simpan tweets dan sentiment yang didapat.
for tweet in searched_tweets:
    cleanedtext = clean_tweet(tweet.text)
    tweet_txt.append(cleanedtext)
    analysis = TextBlob(cleanedtext)
    tweet_stm.append(polarity(analysis.sentiment.polarity))


# Step 6 Mendownload file tweet yang akan disimpan
download_dir = "hasilCrawling.csv"

# Step 7 Membuat DataFrame dan membaca dari file CSV.
df = pd.DataFrame(
    {
        "text": tweet_txt,
        "polarity": tweet_stm
    }
)
df.to_csv(download_dir)

# Step 7 Membaca data dari file CSV
colnames = ['text', 'polarity']
data = pd.read_csv('hasilCrawling.csv')

data_info = data.polarity.value_counts()
l = len(data)
print("Neutral :"+str((data_info['Neutral'] / l) * 100) + " %")
print("Positive :"+str((data_info['Positive'] / l) * 100) + " %")
print("Negative :"+str((data_info['Negative'] / l) * 100) + " %")
