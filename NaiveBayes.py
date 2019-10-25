#!/usr/bin/env python
# coding: utf-8

# Importing the required libraries

# In[1]:


import codecs
from math import log
from collections import Counter


# Toggle for removing stop words. Set remove_stopwords to True to remove stopwords and False to not remove them.

# In[2]:


remove_stopwords = False
stopwords=set()

if remove_stopwords:
    with open("stopwords.txt",'r') as a:
        stopwords = set(a.read().split())
    


# Functions to read dataset and return the priors and word probabilities.

# In[3]:


def laplace_smoothening(w,d,n):
    for i in w:
        if i not in d:
            d[i]=n
        else:
            d[i]+=n
    return d


# In[4]:


def fnames(mypath):
    from os import walk
    f=[]
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        return f


# In[5]:


def read_data(mypath):
    
    sentences =""
    f = fnames(mypath)
    prior=len(f)
    for i in f:
        fin = codecs.open(mypath+i, 'r', encoding='utf-8',
                          errors='ignore')
        data=fin.read().lower() + " "
        sentences+=data
    return prior,sentences.split()


# In[6]:


def get_dataset():
    
    spams, spam_words = read_data("./train/spam/")
    hams, ham_words = read_data("./train/ham/")
    
    if remove_stopwords:
        spam_words = [ x for x in spam_words if x not in stopwords]
        ham_words = [ x for x in ham_words if x not in stopwords]
    
    spam_word_counts = Counter(spam_words)
    ham_word_counts = Counter(ham_words)
    spam_prior = spams/(spams+hams)
    ham_prior = hams/(spams+hams)
    vocab = set(spam_words + ham_words)
    
    #Laplace smoothening
    spam_word_counts = laplace_smoothening(vocab, spam_word_counts, 1)
    ham_word_counts = laplace_smoothening(vocab, ham_word_counts, 1)

    tot_spam_words = sum(spam_word_counts.values())
    spam_word_prob = spam_word_counts
    for i in vocab:
        spam_word_prob[i] = spam_word_prob[i]/tot_spam_words

    tot_ham_words = sum(ham_word_counts.values())
    ham_word_prob = ham_word_counts
    for i in vocab:
        ham_word_prob[i] = ham_word_prob[i]/tot_ham_words

    return spam_prior, ham_prior, spam_word_prob, ham_word_prob, vocab


# Testing the Naive Bayes model

# In[7]:


spam_prior, ham_prior, spam_word_prob, ham_word_prob, vocab = get_dataset()
mypath = "./test/spam/"


X = []
y = []
fl=1
paths = ["./test/spam/"  , "./test/ham/"]
for mypath in paths:
    f = fnames(mypath)
    for i in f:
        fin = codecs.open(mypath+i, 'r', encoding='utf-8',
                        errors='ignore')
        sen = fin.read().lower().split()
        X.append(sen)
        y.append(fl)
    fl-=1

pred = []

for i in X:
    sp = log(spam_prior)
    hp = log(ham_prior)
    for j in i:
        if j in vocab:
            sp += log(spam_word_prob[j])
            hp += log(ham_word_prob[j])
    if sp > hp:
        pred.append(1)
    else:
        pred.append(0)

from sklearn.metrics import accuracy_score
print(accuracy_score(pred, y))


# In[ ]:





# In[ ]:




