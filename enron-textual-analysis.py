# Automatic Text Analysis of Values in the Enron Email Dataset
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
emails = pd.read_csv('emails.csv')
email_subset = emails.sample(frac=0.02, random_state=1)
#email_subset = emails[:10000]
print(email_subset.shape)
print(email_subset.head())


# In[2]:


def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email


# In[3]:


def parse_into_emails(messages):
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'),
        'to': map_to_list(emails, 'to'),
        'from_': map_to_list(emails, 'from')
    }


# In[4]:


def map_to_list(emails, key):
    results = []
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results


# In[5]:


email_df = pd.DataFrame(parse_into_emails(email_subset.message))
print(email_df.head())


# In[6]:


import re
import numpy as np


# In[7]:


import gensim


# In[8]:


import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


# In[9]:


# spacy for lemmatization
import spacy


# In[10]:


# for plotting
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt


# In[11]:


#import nltk
#nltk.download('stopwords')


# In[12]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# In[13]:


print(email_df.iloc[2]['body']) # displays info below


# In[14]:


# Convert email body to list
data = email_df.body.values.tolist()


# In[15]:


# tokenize - break down each sentence into a list of words
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  
        # deacc=True removes punctuations


# In[16]:


data_words = list(sent_to_words(data))


# In[17]:


print(data_words[3])


# In[18]:


from gensim.models.phrases import Phrases, Phraser


# In[19]:


# Build the bigram and trigram models
bigram = Phrases(data_words, min_count=5, threshold=100) 
# higher threshold fewer phrases.
trigram = Phrases(bigram[data_words], threshold=100)


# In[20]:


# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)


# In[21]:


print(trigram_mod[bigram_mod[data_words[200]]])


# In[22]:


# remove stop_words, make bigrams and lemmatize
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[23]:


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)


# In[24]:


# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)


# In[27]:


# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# In[29]:


# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, 
                                allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# In[31]:


print(data_lemmatized[200])


# In[33]:


# create dictionary and corpus both are needed for (LDA) topic modeling

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


# In[35]:


import warnings


# In[37]:


warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[39]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[40]:


# topic modeling
# corpus, dictionary and number of topics required for LDA
# alpha and eta are hyperparameters that affect sparsity of the topics
# chunksize is the number of documents to be used in each training chunk
# update_every determines how often the model parameters should be updated
# passes is the total number of training passes
# Print the Keyword in the 10 topics


# In[41]:


print(lda_model.print_topics())
# The weights reflect how important a keyword is to that topic.


# In[43]:


doc_lda = lda_model[corpus]


# In[44]:


# Model perplexity and topic coherence provide a convenient
# measure to judge how good a given topic model is.
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
# a measure of how good the model is. lower the better.


# In[46]:


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[48]:


# Visualize the topics
pyLDAvis.enable_notebook(sort=True)
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)


# In[49]:


pyLDAvis.display(vis)


# In[109]:


import os
os.environ.update({'MALLET_HOME':r'C:/home/jupyter_projects/mallet-2.0.8/'})


# In[110]:


mallet_path = 'C:\\home\\jupyter_projects\\mallet-2.0.8\\bin\\mallet'


# In[111]:


from gensim.test.utils import common_corpus, common_dictionary


# In[112]:


from gensim.models.wrappers import LdaMallet


# In[113]:


ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)


# In[114]:


# Show Topics
print(ldamallet.show_topics(formatted=False))


# In[115]:


# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


# In[128]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[130]:


# run
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)


# In[132]:


# Show graph
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[134]:


# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[135]:


# Select the model and print the topics
optimal_model = model_list[4]
model_topics = optimal_model.show_topics(formatted=False)
print(optimal_model.print_topics(num_words=10))


# In[136]:


def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


# In[137]:


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)


# In[138]:


# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']


# In[139]:


# Show
df_dominant_topic.head(10)


# In[140]:


df_dominant_topic.Keywords.iloc[1]


# In[141]:


df_dominant_topic.Text.iloc[1]


# In[142]:


# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)


# In[143]:


# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)


# In[144]:


# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]


# In[145]:


# Show
sent_topics_sorteddf_mallet


# In[147]:


import csv


# In[148]:


sent_topics_sorteddf_mallet.to_csv('topics.csv')

