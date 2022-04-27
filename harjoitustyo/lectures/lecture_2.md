# Datan kerääminen

_Niklas Nurminen - Johdanto datatieteeseen_

Valmiin datasetin löysin Kagglesta. Dataset koostui kahdesta eri .csv tiedostosta. Toinen tiedosto oli täynnä aitoja uutisia, kun puolestaan toinen oli täysin väärennetty. Tulen jakamaan datasetin osiin: opetus- ja testidataksi. Datan setin löydät [tästä.](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) Datasetin avulla pystytään opettamaan luokittelija, mikä myöhemmin pystyy arvioimaan, että onko jokin uutinen aito. HUOM. käsitellään vain englannin kielisiä uutisia.

Toinen datasetti kerättiin Twitteristä käyttämällä Pythonin Tweepy kirjastoa. Tällä datasetillä pystyn hyvin tarkastelemaan aiemmin opetetun luokittelijan toimintaa. Mielenkiintoista nähdä, kuinka suuren osan luokittelija luokittelee epäaidoksi uutiseksi. Jos intoa ja aikaa riittää niin myöhemmin voisi vielä kerätä esimerkiksi uutistoimisto BBC twitter kanavalta uutisia. Voisi ainakin kuvitella, että näistä tweettauksista suurin osa olisi aitoja.

```python
import tweepy
import pandas as pd

# Add your Bearer token here.
client = tweepy.Client(bearer_token='')

# Create Pandas dataframe with three columns
df = pd.DataFrame(columns=['text', 'language'])

# Query tweets with #news. Show only 100 tweets
query = '#news'
tweets = client.search_recent_tweets(query=query, tweet_fields=['lang'], max_results=100)

for tweet in tweets.data:
    text = tweet.text
    language = tweet.lang

    # Focus only to tweets in english
    if language == 'en':
        tmp = [text, language]
        df.loc[len(df)] = tmp

# Total amount of tweets in dataframe
print('Amount: ', len(df))

# Show first 5 lines of dataframe
df.head()
```

### Lähteet
1. [Tweepy dokumentaatio](https://docs.tweepy.org/en/stable/)
2. [Twitter developer platform](https://developer.twitter.com/en)
3. [Fake and real news dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### Pros & cons
1. Kagglesta löytyy erittäin paljon valmiita datasettejä.
2. Tweepy kirjaston käyttäminen oli melko yksinkertaista.
3. Kipeenä on vaikea keskittyä tekemiseen.


