#!/usr/bin/env python
# coding: utf-8

# Importing the required packages

# In[1]:


import codecs
from math import exp
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Toggle for removing stop words. Set remove_stopwords to True to remove stopwords and False to not remove them.

# In[2]:


remove_stopwords = True
stopwords=set()

if remove_stopwords:
    with open("stopwords.txt",'r') as a:
        stopwords = set(a.read().split())
    


# Generating vocabulary from the dataset

# In[3]:


def gen_vocab():
    voc = []
    paths = ["./train/ham/", "./train/spam/"]
    for mypath in paths:
        f = fnames(mypath)
        for i in f:
            fin = codecs.open(mypath+i, 'r', encoding='utf-8',
                              errors='ignore')
            sen = (fin.read()).lower().split()
            voc.extend(sen)
    return set(voc)- stopwords


# Assigning an index to each word

# In[4]:


def gen_index():
    vocab = gen_vocab()
    d={}
    k=0
    for i in vocab:
        d[i]=k
        k+=1
    return d,len(vocab)


# Read data and fill dataset

# In[5]:


def fnames(mypath): # returns all file names in a directory
    from os import walk
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        return f


def createDataset(s="train"):
    words = []
    paths= ["./{}/spam/".format(s),"./{}/ham/".format(s)]
    flag=1
    features=[]
    labels=[]
    l=[]
    for mypath in paths:
        files = fnames(mypath)
        for file in files:
            X=[1] # Inserting X0
            X += [0 for i in range(voc_len)] # Inserting X1,X2 ...Xn
            fin = codecs.open(mypath+file, 'r', encoding='utf-8',errors='ignore')
            data = fin.read().lower().split()
            for word in data:
                try:
                    X[d[word]]+=1
                except:
                    pass # required during testing for words in test data but not in train data
            labels.append(flag)
            features.append(X)
        flag-=1
    X=np.asarray(features)
    y=np.asarray(labels)
    return X,y


# Call functions to generate train and test data

# In[6]:


vocab = gen_vocab()
d,voc_len = gen_index()
X_train,y_train = createDataset("train")
X_test,y_test = createDataset("test")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Functions to train Linear Regression model and predict function

# In[7]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[8]:


def trainLR(X, y, num_iter, lr, lam):
    weights = np.zeros(X.shape[1])

    for i in range(num_iter):
        z = X@weights
        h = sigmoid(z)
        m = y.size
        gradient = 1/m * (X.T@(h - y) + lam/(2*m)*weights.T@weights)
        weights -= lr * gradient
        z = X@weights
        h = sigmoid(z)
    return weights


# In[9]:


def predict(X):
    return sigmoid(np.dot(X, weights))>0.5


# Training a Linear regression and printing the accuracy score

# In[10]:


num_of_iter = 1000
learning_rate = 0.25
lam = 0.1
weights = trainLR(X_train, y_train, num_of_iter, learning_rate, lam)

y_pred = predict(X_test)
print("Testing accuracy : ",(accuracy_score(y_test, y_pred)))


# In[ ]:




