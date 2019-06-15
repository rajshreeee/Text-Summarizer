

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


# In[3]:


clean_summaries = __loadStuff("./data/clean_summaries.p")
clean_texts = __loadStuff("./data/clean_texts.p")

sorted_summaries = __loadStuff("./data/sorted_summaries.p")
sorted_texts = __loadStuff("./data/sorted_texts.p")
word_embedding_matrix = __loadStuff("./data/word_embedding_matrix.p")

vocab_to_int = __loadStuff("./data/vocab_to_int.p")
int_to_vocab = __loadStuff("./data/int_to_vocab.p")






# ## 2. Preparing the Data

# In[15]:


# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}


# In[16]:


def clean_text(text, remove_stopwords = True):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    
    # Convert words to lower case
    text = text.lower()
    
    # Replace contractions with their longer forms 
    if True:
        # We are not using "text.split()" here
        #since it is not fool proof, e.g. words followed by punctuations "Are you kidding?I think you aren't."
        text = re.findall(r"[\w']+", text)
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)# remove links
    text = re.sub(r'\<a href', ' ', text)# remove html link tag
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text


# In[17]:


clean_text("That's a great movie,Can you believe it?I've.But you may not.")


# ### Clean the summaries and texts
# We will remove the stopwords from the texts because they do not provide much use for training our model. However, we will keep them for our summaries so that they sound more like natural phrases. 

# In[14]:


clean_summaries = []
for summary in reviews.Summary:
    clean_summaries.append(clean_text(summary, remove_stopwords=False))
print("Summaries are complete.")

clean_texts = []
for text in reviews.Text:
    clean_texts.append(clean_text(text))
print("Texts are complete.")


# In[15]:


# Inspect the cleaned summaries and texts to ensure they have been cleaned well
for i in range(5):
    print("Clean Review #",i+1)
    print(clean_summaries[i])
    print(clean_texts[i])
    print()


# ### Count the number of occurrences of each word in a set of text

# In[16]:


def count_words(count_dict, text):
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1


# #### Give the function a try

# In[17]:


mydict = {}
count_words(mydict, ["that is a great great great dog","you have a great dog"])
mydict


# In[18]:


word_counts = {}
count_words(word_counts, clean_summaries)
count_words(word_counts, clean_texts)
print("Size of Vocabulary:", len(word_counts))


# Let's see how may "hero" occurs in the data

# In[19]:


word_counts["hero"]


# In[20]:



embeddings_index = {}
with open('./model/numberbatch-en-17.06.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings:', len(embeddings_index))


# ### Take a look at the CN embedding dimension

# In[21]:


embeddings_index["hero"].shape


# ### Find the number of words that are missing from CN, and are used more than our threshold.
# 
# I use a **threshold** of 20, so that words not in CN can be added to our **word_embedding_matrix**, but they need to be common enough in the reviews so that the model can understand their meaning.

# In[22]:


missing_words = 0
threshold = 20

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1
            
missing_ratio = round(missing_words/len(word_counts),4)*100
            
print("Number of words missing from CN:", missing_words)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))


# ### What are those missing words in the CN
# Looks mostly products' brand.

# In[23]:


missing_words = []
for word, count in word_counts.items():
    if count > threshold and word not in embeddings_index:
        missing_words.append((word,count))
missing_words[:30]


# ### Words to indexes, indexes to words dicts
# Limit the vocab that we will use to words that appear ≥ threshold or are in CN

# In[24]:


#dictionary to convert words to integers
vocab_to_int = {} 
# Index words from 0
value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

print("Total number of unique words:", len(word_counts))
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(usage_ratio))


# ### Create word embedding matrix
# It has shape (nb_words, embedding_dim) i.e. (59072, 300) in this case. 1st dim is word index, 2nd dim is from CN or random generated.

# In[25]:


# Need to use 300 for embedding dimensions to match CN's vectors.
embedding_dim = 300
nb_words = len(vocab_to_int)

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in CN, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))


# ### Function to convert sentences to sequence of words indexes
# It also use `<UNK>` index to replace unknown words, append `<EOS>` (End of Sentence) to the sequences if eos is set True

# In[26]:


def convert_to_ints(text, word_count, unk_count, eos=False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count


# Apply convert_to_ints to clean_summaries and clean_texts

# In[27]:



word_count = 0
unk_count = 0

int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)

unk_percent = round(unk_count/word_count,4)*100

print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))


# ### Take a look at what the sequence looks like
# Each number here represents a word

# In[28]:


int_summaries[:3]


# ### Function to get the length of each sequence

# In[29]:


def create_lengths(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])


# In[30]:


create_lengths(int_summaries[:3])


# Get statistic summary of the length of summaries and texts

# In[31]:


lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)

print("Summaries:")
print(lengths_summaries.describe())
print()
print("Texts:")
print(lengths_texts.describe())


# ### See what's the max squence length we can cover by percentile

# In[32]:


# Inspect the length of texts
print(np.percentile(lengths_texts.counts, 89.5))
print(np.percentile(lengths_texts.counts, 95))
print(np.percentile(lengths_texts.counts, 99))


# In[33]:


# Inspect the length of summaries
print(np.percentile(lengths_summaries.counts, 90))
print(np.percentile(lengths_summaries.counts, 95))
print(np.percentile(lengths_summaries.counts, 99))


# ## Function to counts the number of time `<UNK>` appears in a sentence

# In[34]:


def unk_counter(sentence):
    '''Counts the number of time UNK appears in a sentence.'''
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
    return unk_count


# **Filter** for length limit and number of `<UNK>`s
# 
# **Sort** the summaries and texts by the length of the element in **texts** from shortest to longest
# 

# In[39]:


max_text_length = 83 # This will cover up to 89.5% lengthes
max_summary_length = 13 # This will cover up to 99% lengthes
min_length = 2
unk_text_limit = 1 # text can contain up to 1 UNK word
unk_summary_limit = 0 # Summary should not contain any UNK word

def filter_condition(item):
    int_summary = item[0]
    int_text = item[1]
    if(len(int_summary) >= min_length and 
       len(int_summary) <= max_summary_length and 
       len(int_text) >= min_length and 
       len(int_text) <= max_text_length and 
       unk_counter(int_summary) <= unk_summary_limit and 
       unk_counter(int_text) <= unk_text_limit):
        return True
    else:
        return False

int_text_summaries = list(zip(int_summaries , int_texts))
int_text_summaries_filtered = list(filter(filter_condition, int_text_summaries))
sorted_int_text_summaries = sorted(int_text_summaries_filtered, key=lambda item: len(item[1]))
sorted_int_text_summaries = list(zip(*sorted_int_text_summaries))
sorted_summaries = list(sorted_int_text_summaries[0])
sorted_texts = list(sorted_int_text_summaries[1])
# Delete those temporary varaibles
del int_text_summaries, sorted_int_text_summaries, int_text_summaries_filtered
# Compare lengths to ensure they match
print(len(sorted_summaries))
print(len(sorted_texts))


# ### Inspect the length of text in sorted_texts

# In[40]:


lengths_texts = [len(text) for text in sorted_texts]
lengths_texts[:20]


# ## Save data for later

# In[41]:


__pickleStuff("./data/clean_summaries.p",clean_summaries)
__pickleStuff("./data/clean_texts.p",clean_texts)

__pickleStuff("./data/sorted_summaries.p",sorted_summaries)
__pickleStuff("./data/sorted_texts.p",sorted_texts)
__pickleStuff("./data/word_embedding_matrix.p",word_embedding_matrix)

__pickleStuff("./data/vocab_to_int.p",vocab_to_int)
__pickleStuff("./data/int_to_vocab.p",int_to_vocab)


