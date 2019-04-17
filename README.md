Text-Summarizer : a model that creates reviews related to fine food.

Dataset:The dataset has been obtained from https://www.kaggle.com/snap/amazon-fine-food-reviews/. 
It includes: 568,454 reviews, 256,059 users and 4,258 products.

To build our model we will use a two-layered bidirectional RNN with LSTMs on the input data and two layers, 
each with an LSTM using bahdanau attention on the target data.

Dependencies: Tensorflow, numpy, pandas,nltk, python

Inspired by:https://github.com/Tony607/Summarizing_Text_Amazon_Reviews
