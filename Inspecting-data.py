
# coding: utf-8

# # Summarizing Text with Amazon Reviews

# The objective of this project is to build a model that can create relevant summaries for reviews written about fine foods sold on Amazon. This dataset contains above 500,000 reviews, and is hosted on [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews).
# 
# To build our model we will use a two-layered bidirectional RNN with LSTMs on the input data and two layers, each with an LSTM using bahdanau attention on the target data.
# 
# The sections of this project are:
# - [1.Inspecting the Data](#1.-Insepcting-the-Data)
# - [2.Preparing the Data](#2.-Preparing-the-Data)
# - [3.Building the Model](#3.-Building-the-Model)
# - [4.Training the Model](#4.-Training-the-Model)
# - [5.Making Our Own Summaries](#5.-Making-Our-Own-Summaries)
# 
# ## Download data
# Amazon Reviews Data: [Reviews.csv](https://www.kaggle.com/snap/amazon-fine-food-reviews/downloads/Reviews.csv)
# 
# word embeddings [numberbatch-en-17.06.txt.gz](https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-en-17.06.txt.gz)
# after download, extract to **./model/numberbatch-en-17.06.txt**

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
print('TensorFlow Version: {}'.format(tf.__version__))


# In[2]:


import pickle
def __pickleStuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()
def __loadStuff(filename):
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


# ## Load those prepared data and skip to section "[3. Building the Model](#3.-Building-the-Model)"
# Once we have run through the "[2.Preparing the Data](#2.-Preparing-the-Data)" section, we should have those data, uncomment and run those lines.

# In[3]:


clean_summaries = __loadStuff("./data/clean_summaries.p")
clean_texts = __loadStuff("./data/clean_texts.p")

sorted_summaries = __loadStuff("./data/sorted_summaries.p")
sorted_texts = __loadStuff("./data/sorted_texts.p")
word_embedding_matrix = __loadStuff("./data/word_embedding_matrix.p")

vocab_to_int = __loadStuff("./data/vocab_to_int.p")
int_to_vocab = __loadStuff("./data/int_to_vocab.p")


# ## 1. Insepcting the Data

# In[3]:


reviews = pd.read_csv("Reviews.csv")


# In[4]:


reviews.shape


# In[5]:


reviews.head()


# In[6]:


# Check for any nulls values
reviews.isnull().sum()


# In[7]:


# Remove null values and unneeded features
reviews = reviews.dropna()
reviews = reviews.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator',
                        'Score','Time'], 1)
reviews = reviews.reset_index(drop=True)


# In[8]:


reviews.shape


# In[9]:


reviews.head()


# In[10]:


# Inspecting some of the reviews
for i in range(5):
    print("Review #",i+1)
    print(reviews.Summary[i])
    print(reviews.Text[i])
    print()


