#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
import textblob
import sklearn


# In[2]:


train = pd.read_csv('EmailsSentiment1.csv')


# In[3]:


train.head()


# In[4]:


## preprocess
train['word_count'] = train['body'].apply(lambda x: len(str(x).split(" ")))
train[['body','word_count']].head()


# In[5]:


train['char_count'] = train['body'].str.len() ## this also includes spaces
train[['body','char_count']].head()


# In[6]:


def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

train['avg_word'] = train['body'].apply(lambda x: avg_word(x))
train[['body','avg_word']].head()


# In[7]:


from nltk.corpus import stopwords
stop = stopwords.words('english')

train['stopwords'] = train['body'].apply(lambda x: len([x for x in x.split() if x in stop]))
train[['body','stopwords']].head()


# In[8]:


#hastags
train['hastags'] = train['body'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
train[['body','hastags']].head()


# In[9]:


#$
train['hastags'] = train['body'].apply(lambda x: len([x for x in x.split() if x.startswith('$')]))
train[['body','hastags']].head()


# In[10]:


## numerics
train['numerics'] = train['body'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train[['body','numerics']].head()


# In[11]:


## uppercase
train['upper'] = train['body'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['body','upper']].head()


# In[12]:


## Process
train['body'] = train['body'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['body'].head()


# In[13]:


## remove punctuation
train['body'] = train['body'].str.replace('[^\w\s]','')
train['body'].head()


# In[14]:


## remove stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['body'] = train['body'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['body'].head()


# In[15]:


freq = pd.Series(' '.join(train['body']).split()).value_counts()[:10]
freq


# In[16]:


##freq = list(freq.index)
##train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
##train['tweet'].head()


# In[17]:


rare = pd.Series(' '.join(train['body']).split()).value_counts()[-10:]
rare


# In[18]:


## spelling correction
from textblob import TextBlob
train['body'][:5].apply(lambda x: str(TextBlob(x).correct()))


# In[ ]:


## tokenization


# In[ ]:


TextBlob(train['body'][1]).words


# In[ ]:


## stemming


# In[ ]:


from nltk.stem import PorterStemmer
st = PorterStemmer()
train['body'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# In[ ]:


## lemmatization


# In[ ]:


from textblob import Word
train['body'] = train['body'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train['body'].head()


# In[ ]:


## N-grams


# In[ ]:


TextBlob(train['body'][0]).ngrams(2)


# In[ ]:


TextBlob(train['body'][0]).ngrams(3)


# In[ ]:


## term frequency


# In[ ]:


tf1 = (train['body'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
##tf1


# In[ ]:


## inverse document frequency
##The intuition behind inverse document frequency (IDF) is that a word is not of much use to us 
##if it’s appearing in all the documents.

##Therefore, the IDF of each word is the log of the ratio of the total number of rows to the number of rows 
##in which that word is present.

##IDF = log(N/n), where, N is the total number of rows and n is the number of rows in which the word was present.


# In[ ]:


for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['body'].str.contains(word)])))

##tf1


# In[ ]:


## Term Frequency – Inverse Document Frequency (TF-IDF)


# In[ ]:


tf1['tfidf'] = tf1['tf'] * tf1['idf']
##tf1


# In[ ]:


## sklearn


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(train['body'])


# In[ ]:


##train_vect


# In[ ]:


## bag of words


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(train['body'])


# In[ ]:


##train_bow


# In[ ]:


## sentiment analysis
## before applying any ML/DL models (which can have a separate feature detecting the sentiment using the textblob library), 
##let’s check the sentiment of the first few tweets.


# In[ ]:


train['body'][:5].apply(lambda x: TextBlob(x).sentiment)


# In[ ]:


##Above, you can see that it returns a tuple representing polarity and subjectivity of each tweet. 
##Here, we only extract polarity as it indicates the sentiment as value nearer to 1 means a positive sentiment 
##and values nearer to -1 means a negative sentiment. This can also work as a feature for building a machine learning model.


# In[ ]:


train['sentiment'] = train['body'].apply(lambda x: TextBlob(x).sentiment[0] )
train[['body','sentiment']].head()


# In[ ]:


## word embeddings


# In[ ]:


from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'


# In[ ]:


df = pd.DataFrame(train)


# In[ ]:


df.head()


# In[ ]:


text_sample = TextBlob("Don’t know if it’s the Amazon brand laminator or the sheets but corners consistently fold over in the machine. The sheets are fresh out of the box and package and fed completely flat through and still they come out folded. And tons of tiny bubbles! Maybe that normal but I had higher expectations. (PSA!!! Do not try and laminate concert tickets because they will burn/get completely ruined!")


# In[ ]:


text_sample.sentiment

