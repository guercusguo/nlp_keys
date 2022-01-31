#!/usr/bin/env python
# coding: utf-8

# pandas export
## df.to_csv('file_name.csv') #default export
## df.to_csv('file_name.csv', index=False) #no index
## dt.to_csv('file_name.csv',header=False) #no column names
## df.to_csv('file_name.csv', encoding='utf-8') #force utf-8
## dt.to_csv('file_name.csv',sep='\t') # custom delimiter (e.g. tab)
## dt.to_csv('file_name.csv',na_rep='Unkown') # missing value save as Unknown
## dt.to_csv('file_name.csv',float_format='%.2f') # rounded to two decimals
## dt.to_csv('file_name.csv',columns=['name']) # export certain columns

# ## Path check


import os
from os import path
path = os.getcwd()
print(path)


# ## Libraries import
# Main: NLTK, SPACY (pre-pro), PANDAS (dataframe), SCIKIT-learn
# ***Notes***
# #WordCloud -> dead kernel on Win10
# #CountVectorizer -> on corpus (list, tuples) -> adapt on PD


#libraries for dataframes
import pandas as pd
from collections import Counter
from itertools import chain


## NLTK (import + download/update)
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import everygrams
from nltk import sent_tokenize


import pandas as pd
import gensim
import spacy
from tqdm import tqdm


from sklearn.feature_extraction.text import CountVectorizer


from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


## SCIKIT-Learn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# RAKE and utilities
import RAKE
import operator
import numpy as np


from scipy.sparse import coo_matrix


# ## Data load
lectures = pd.read_csv("CHAP_CSVOPT_2.csv", header=0)
import csv
keys_str =[]
with open('glossario_trilingue_clean_lower.csv', newline='') as f:
    reader = csv.reader(f)
    keys_glos = list(reader)
    keys_str = str(keys_glos)
glos_keys = pd.read_csv ('glossario_trilingue_clean_lower.csv', header=0)

# ## Reference Keywords Tokenization
# #### NLTK -> everygram (1-4)
from nltk import everygrams
def add_ngram_col (keywords):
    tokens = nltk.word_tokenize(keywords)
    if len(tokens) == 0:
        zero_length_token='zerolengthunk'
        return [zero_length_token]
    else:
        ngram = []
        if len(tokens) == 1:
            ngram.extend(tokens)
        if len(tokens) == 2:
            ngram.extend([f'{w0} {w1}' for w0, w1 in nltk.bigrams(tokens)])
        if len(tokens) == 3:
            ngram.extend([f'{w0} {w1} {w2}' for w0, w1, w2 in nltk.trigrams(tokens)])
        if len(tokens) == 4:
            ngram.extend([f'{w0} {w1} {w2} {w3}' for w0, w1, w2, w3 in nltk.everygrams(tokens, min_len=4, max_len=4)])
        if len(tokens) == 5:
            ngram.extend([f'{w0} {w1} {w2} {w3} {w4}' for w0, w1, w2, w3, w4 in nltk.everygrams(tokens, min_len=5, max_len=5)])
    return str(ngram)

glos_keys['ngram'] = np.vectorize(add_ngram_col)(glos_keys['keyword'])
glos_keys_list = (glos_keys['keyword'])
print (glos_keys_list)
glos_keys_list.head()
glos_keys.head()


#  Preliminary statistics on keywords
glos_keys.describe()
glos_keys.to_csv('glos_keys.csv')

# # Preliminary statistics on dataset
lectures.head()
lectures['word_count'] = lectures['text'].apply(lambda x: len(str(x).split(" ")))
lectures[['subject','text','word_count']].head()
lectures.word_count.describe()
freq = pd.Series(' '.join(lectures['text']).split()).value_counts()[:10]
freq
#Identify uncommon words
freq1 =  pd.Series(' '.join(lectures
         ['text']).split()).value_counts()[-10:]
freq1

# ## Data normalization
# ## Stopwords removal (dataset + keywords)
# ### default stopwords from NLTK + most frequent words of raw dataset (if not present in default)
##Creating a list of stop words and adding custom stopwords
stop_words = set(stopwords.words("english"))
##Creating a list of custom stopwords
new_words = ["the", "of", "to", "and", "a", "in", "that", "The", "with", "as", "are","on", "or", "by", "for", "which", "be", "an", "it","this","therefore","case","each","its","not","between", "can", "this", "In", "The"]
stop_words = stop_words.union(new_words)
print(stop_words)

# #Creating a list of stop words and adding custom stopwords
stop_words_keys = set(stopwords.words("english"))

# #Creating a list of custom stopwords
new_words_keys = ["the", "of", "to", "and", "a", "in", "that", "The", "with", "as", "are","on", "or", "by", "for", "which", "be", "an", "it","this","therefore","case","each","its","not","between", "can", "this", "In", "The"]
stop_words_keys = stop_words_keys.union(new_words)
print(stop_words_keys)


# ## normalised corpus (list) for dataset and keywords
# - punctuation removal
# - lowercase conversion
# - tags removal (for tag based languages)
# - special characters removal (through regex)+
# - lemmatisation
#
# Lemmatisation: is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form


corpus = []
for i in range(0, 125):
    #Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', lectures['text'][i])

    #Convert to lowercase
    text = text.lower()

    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)

    ##Convert to list from string
    text = text.split()

    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in
            stop_words]
    text = " ".join(text)
    corpus.append(text)

lectures['clean_text'] = corpus

corpus_keys = []
for i in range(0, 862):
    #Remove punctuations
    keys = re.sub('[^a-zA-Z]', ' ', glos_keys['keyword'][i])

    #Convert to lowercase
    keys = keys.lower()
    keys.lower()

    #remove tags
    keys=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",keys)

    # remove special characters and digits
    keys=re.sub("(\\d|\\W)+"," ",keys)

    ##Convert to list from string
    keys = keys.split()

    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in
            stop_words]
    keys = " ".join(keys)
    corpus_keys.append(keys)
corpus_keys[14]
corpus_keys.describe
#View corpus item
corpus[0]
# ## Preliminary visualization of data properties

# # WordCloud -> convert to python for macOS. Not working on W10
# wordcloud = WordCloud(
#                           background_color='white',
#                           stopwords=stop_words,
#                           max_words=100,
#                           max_font_size=50,
#                           random_state=42
#                          ).generate(str(corpus[11]))
# print(wordcloud)
# fig = plt.figure(1)
# plt.imshow(wordcloud)
# plt.axis('off')
# plt.show()
# fig.savefig("word1.png", dpi=900)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stop_words,
                          max_words=100,
                          max_font_size=50,
                          random_state=42
                         ).generate(str(corpus[1]))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)


# # Word vectorialization using SciKit CountVectorizer
#
# ### Convert a collection of text documents to a matrix of token counts. This implementation produces a sparse representation of the counts using scipy.sparse.csr_matrix.
#
# cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
#
# **max_df** — When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). This is to ensure that we only have words relevant to the context and not commonly used words.
#
# **max_features** — determines the number of columns in the matrix.
#
# **n-gram range** — we would want to look at a list of single words, two words (bi-grams) and three words (tri-gram) combinations.
#
# ### set max_df (default = 0.8), set max_features (default = 10.000), set n-gram range (default = 1-3)
#
# max_df ->  lower for lectures (14 documents), higher for chapters (110 documents).
# max_features -> based on number of tokens
# min_df -> not set. Should?
#
cv=CountVectorizer(max_df=0.9,stop_words=stop_words, max_features=20000, ngram_range=(1,3))
X=cv.fit_transform(corpus)
list(cv.vocabulary_.keys())[:10]
cv_keys=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=20000, ngram_range=(1,3))
Y=cv_keys.fit_transform(corpus_keys)
list(cv_keys.vocabulary_.keys())[:10]

# ## Most occurring uni, bi, tri grams (histograms)

# ### Uni-grams (corpus + corpus_keys)
#Most frequently occuring words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                       reverse=True)
    return words_freq[:n]

#Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus, n=40)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word", "Freq"]

#Barplot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)

#Most frequently occuring keywords
def get_top_n_words(corpus_keys, n=None):
    vec = CountVectorizer().fit(corpus_keys)
    bag_of_words = vec.transform(corpus_keys)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                       reverse=True)
    return words_freq[:n]

#Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus_keys, n=10)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word", "Freq"]

#Barplot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
sns.set_style(style="whitegrid")
sns.color_palette("pastel")
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)

# ### Bi-grams (corpus + corpus_keys)
#Most frequently occuring Bi-grams
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                reverse=True)
    return words_freq[:n]

top2_words = get_top_n2_words(corpus, n=40)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
print(top2_df)

#Barplot of most freq Bi-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=45)

#Most frequently occuring Bi-grams
def get_top_n2_words(corpus_keys, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),
            max_features=2000).fit(corpus_keys)
    bag_of_words = vec1.transform(corpus_keys)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                reverse=True)
    return words_freq[:n]

top2_words = get_top_n2_words(corpus_keys, n=10)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
print(top2_df)

#Barplot of most freq Bi-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=45)


# ### Tri-grams (corpus + corpus_keys)
#Most frequently occuring Tri-grams
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3),
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                reverse=True)
    return words_freq[:n]

top3_words = get_top_n3_words(corpus, n=10)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]
print(top3_df)

#Barplot of most freq Tri-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
j.set_xticklabels(j.get_xticklabels(), rotation=45)

#Most frequently occuring Tri-grams
def get_top_n3_words(corpus_keys, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3),
           max_features=2000).fit(corpus_keys)
    bag_of_words = vec1.transform(corpus_keys)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                reverse=True)
    return words_freq[:n]

top3_words = get_top_n3_words(corpus_keys, n=50)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]
print(top3_df)

#Barplot of most freq Tri-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
j.set_xticklabels(j.get_xticklabels(), rotation=45)

lectures['test_keys'] = [', '.join(set([y for y in x.split() if y in cv_keys.vocabulary_.keys()])) for x in lectures['clean_text']]
lectures.head()
sentences = []
for row in lectures.itertuples():
    for sentence in sent_tokenize(row[8]):
        sentences.append((row[1], row[2], sentence))
lec_sent = pd.DataFrame(sentences, columns=['number','subject', 'text'])
lec_sent['ch_length'] = lec_sent['text'].str.len()
lec_sent.head()
lec_small = lec_sent[lec_sent.ch_length < 66]
lec_small


# minimum = "^\(\d+\)|^\*+" #Pattern to identify string starting with (number),*,**.
#
# #print(df)
# #Selecting index based on the above pattern
# selected_index = lec_sent[lec_sent["ch_length"].index.values
#
# for index in selected_index:
#     i=1
#     #Merging row until next selected index found and add merged rows to delete_index list
#     while(index+i not in selected_index and index+i < len(lec_sent)-1):
#         lec_sent.at[index, 'text'] += ' ' + df.at[index+i, 'text']
#         delete_index.append(index+i)
#         i+=1
#
# lec_sent.drop(delete_index,inplace=True)
lec_sent.describe()
lec_sent.to_csv('lec_sent.csv')
lectures.iloc[5]['test_keys']

def find_keywords(lecture, key_set):
    return list(zip(*[lecture[i:] for i in range (key_set)]))

lectures['unigrams'] = lectures['clean_text'].map(lambda x: find_keywords(x.split(" "), 1))
lectures['bigrams'] = lectures['clean_text'].map(lambda x: find_keywords(x.split(" "), 2))
lectures['trigrams'] = lectures['clean_text'].map(lambda x: find_keywords(x.split(" "), 3))
lectures['tetragrams'] = lectures['clean_text'].map(lambda x: find_keywords(x.split(" "), 4))
lectures.head()

lectures['unigrams_match'] = lectures['unigrams'].apply(lambda x : [j for j in glos_keys['keyword'] if j in  [' '.join(i) for i in x]])
lectures['bigrams_match'] = lectures['bigrams'].apply(lambda x : [j for j in glos_keys['keyword'] if j in  [' '.join(i) for i in x]])
lectures['trigrams_match'] = lectures['trigrams'].apply(lambda x : [j for j in glos_keys['keyword'] if j in  [' '.join(i) for i in x]])
lectures['tetragrams_match'] = lectures['tetragrams'].apply(lambda x : [j for j in glos_keys['keyword'] if j in  [' '.join(i) for i in x]])
lectures['unigrams_match_count'] = lectures['unigrams_match'].apply(lambda x: len(list(x)))
lectures['bigrams_match_count'] = lectures['bigrams_match'].apply(lambda x: len(list(x)))
lectures['trigrams_match_count'] = lectures['trigrams_match'].apply(lambda x: len(list(x)))
lectures['tetragrams_match_count'] = lectures['tetragrams_match'].apply(lambda x: len(list(x)))
lectures.head()

#lectures['zero_count'] = (lectures['trigrams_match'].values == '').sum()
#lectures['zero_count'] = len(lectures[lectures['trigrams_match'] == 'geographic information system'])
#df.loc[df['currency'] == ''].count().iloc[0]
#lectures['zero_count'] = lectures.trigrams_count.apply(lambda c: c==[])
#lectures['zero_count']=lectures.loc[(lectures['trigrams_match'].str.len() == 0),:]
#lectures['zero_count'] = lectures.mask(lectures.applymap(str).eq('[]'))
#lectures.applymap(str).eq('[]')
#lectures.trigrams_match = lectures.trigrams_match.apply(lambda y: np.nan if len(y)==0 else y)

lectures.head()

lectures_dict = lectures
lectures_dict['unigrams_match'] = lectures_dict['unigrams'].apply(lambda x : [j for j in cv_keys.vocabulary_.keys() if j in  [' '.join(i) for i in x]])
lectures_dict['bigrams_match'] = lectures_dict['bigrams'].apply(lambda x : [j for j in cv_keys.vocabulary_.keys() if j in  [' '.join(i) for i in x]])
lectures_dict['trigrams_match'] = lectures_dict['trigrams'].apply(lambda x : [j for j in cv_keys.vocabulary_.keys() if j in  [' '.join(i) for i in x]])
lectures_dict['tetragrams_match'] = lectures_dict['tetragrams'].apply(lambda x : [j for j in cv_keys.vocabulary_.keys() if j in  [' '.join(i) for i in x]])
lectures_dict['unigrams_count'] = lectures_dict['unigrams_match'].apply(lambda x: len(list(x)))
lectures_dict['bigrams_count'] = lectures_dict['bigrams_match'].apply(lambda x: len(list(x)))
lectures_dict['trigrams_count'] = lectures_dict['trigrams_match'].apply(lambda x: len(list(x)))
lectures_dict['tetragrams_count'] = lectures_dict['tetragrams_match'].apply(lambda x: len(list(x)))
lectures.head()
lectures.describe()
lectures_dict.describe()
