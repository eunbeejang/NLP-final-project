
# coding: utf-8

# In[1]:


# For Data Preparation
import tensorflow as tf
import numpy as np
import pandas as pd
import re # regular expressions


# To clean up texts
import nltk.data
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('punkt')
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')


# For Word Embedding
from collections import Counter
import gensim
import gensim.models as g
from gensim.models import Word2Vec
from gensim.models import Phrases

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt


# For the Model
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import LSTM, Bidirectional,Dropout, Input, SpatialDropout1D, CuDNNLSTM, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from mlxtend.plotting import plot_learning_curves
import matplotlib.pyplot as plt
from mlxtend.data import iris_data
from mlxtend.preprocessing import shuffle_arrays_unison

import logging

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


# In[2]:


IMDB_train = pd.read_csv('./IMDB-train.txt', sep='\t', encoding='latin-1', header=None)
IMDB_train_y = IMDB_train[:][1]
IMDB_valid = pd.read_csv('./IMDB-valid.txt', sep='\t', encoding='latin-1', header=None)
IMDB_valid_y = IMDB_valid[:][1]
IMDB_test = pd.read_csv('./IMDB-test.txt', sep='\t', encoding='latin-1', header=None)
IMDB_test_y = IMDB_test[:][1]
stemmer = SnowballStemmer("english", ignore_stopwords=True)

print("Data loaded.")


# In[3]:


def preprocessing(data):
    new_data = []
    #i = 0
    for sentence in (data[:][0]):
        #clean = re.compile('<.*?>')
        new_sentence = re.sub('<.*?>', '', sentence) # remove HTML tags
        new_sentence = re.sub(r'[^\w\s]', '', new_sentence) # remove punctuation
        new_sentence = new_sentence.lower() # convert to lower case
        if new_sentence != '':
            new_data.append(new_sentence)
    return new_data


# In[4]:


frames = [IMDB_train, IMDB_valid]
frames_y = [IMDB_train_y, IMDB_valid_y]
IMDB_train = pd.concat(frames)
IMDB_train_y = pd.concat(frames_y)
IMDB_train=preprocessing(IMDB_train)
IMDB_test=preprocessing(IMDB_test)


# In[5]:


# IMDB_train[0]


# In[6]:


# Convert a sentence into a list of words
def sentence_to_wordlist(sentence, remove_stopwords=False):
    # Convert words to lower case and split them
    words = sentence.lower().split()
    # Lemmatizing
    #words = [lemmatizer.lemmatize(word) for word in words]
    # 6. Return a list of words
    return(words)


# In[7]:


# whole data into a list of sentences where each sentence is a list of word items
def list_of_sentences(data):
    sentences = []
    for i in data:
        sentences.append(sentence_to_wordlist(i))
    return sentences


# In[8]:


train_x = list_of_sentences(IMDB_train)
train_y = IMDB_train_y.tolist()


# In[9]:


# Create Word Vectors

wv_model = Word2Vec(size=128, window=5, min_count=4, workers=4)

wv_model.build_vocab(train_x) 
wv_model.train(train_x, total_examples=wv_model.corpus_count, epochs=wv_model.iter)
word_vectors = wv_model.wv
words = list(wv_model.wv.vocab)

# Calling init_sims will make the model will be better for memory
# if we don't want to train the model over and over again
wv_model.init_sims(replace=True)

#n_words = print(len(words))

print("Number of word vectors: {}".format(len(word_vectors.vocab)))

# save model
wv_model.wv.save_word2vec_format('model.txt', binary=False)

# load model
#new_model = Word2Vec.load('model.bin')


# In[10]:


from gensim.models.keyedvectors import KeyedVectors

new_model = KeyedVectors.load_word2vec_format('model.txt')
#model.save_word2vec_format('model.txt', binary=False)


# In[11]:


original_model = KeyedVectors.load_word2vec_format('model.txt')
retrofitted_model = KeyedVectors.load_word2vec_format('out_vec.txt')


# In[12]:


new_words = list(retrofitted_model.wv.vocab)


# In[13]:


# Build dictionary & inv_vocab

def create_vocab(data_collect, max_vocab):
    # Get raw data
    x_list = data_collect
    sample_count = sum([len(x) for x in x_list])
    words = []
    for data in x_list:
        words.extend([data])
    count = Counter(words) # word count
    inv_vocab = [x[0] for x in count.most_common(max_vocab)]
    vocab = {x: i for i, x in enumerate(inv_vocab, 1)}
    return vocab, inv_vocab


# In[14]:


vocab, inv_vocab = create_vocab(words, len(words))
ret_vocab, ret_inv_vocab = create_vocab(new_words, len(new_words))


# In[15]:


# Find the max length sentence
def find_max_length_sentence(sentence):
    max_length = 0
    for i in sentence:
        length = len(sentence_to_wordlist(i))
        if max_length < length:
            max_length = length
    return max_length


# In[16]:


seq_length = find_max_length_sentence(IMDB_train)


# In[17]:


# Map each word to corresponding vector
def map_to_vec(word):
    vec = wv_model[word]
    return vec


# In[18]:


# Embedding Matrix
def make_emb_matrix(inv_vocab):
    emb_matrix = []
    for word in inv_vocab:
        emb_matrix.append(map_to_vec(word))
    return emb_matrix


# In[19]:


embedding = np.asarray(make_emb_matrix(inv_vocab))
ret_embedding = np.asarray(make_emb_matrix(ret_inv_vocab))


# In[20]:


wv_dim = 100
num_words = len(word_vectors.vocab)
vocab = Counter(words)
ret_vocab = Counter(new_words)


# In[21]:


word_index = {t[0]: i+1 for i,t in enumerate(vocab.most_common(num_words-1))}

train_sequences = [[word_index.get(t, 0) for t in sentence]
             for sentence in IMDB_train[:len(IMDB_train)]]

test_sequences = [[word_index.get(t, 0)
                   for t in sentence] for sentence in IMDB_test[:len(IMDB_test)]]

# Pad zeros to match the size of matrix
train_data = pad_sequences(train_sequences, maxlen=seq_length, padding="post", truncating="post")
test_data = pad_sequences(test_sequences, maxlen=seq_length, padding="post", truncating="post")


# In[22]:


# Initialize the matrix with random numbers
wv_matrix = (np.random.rand(num_words, wv_dim) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= num_words:
        continue
    try:
        embedding_vector = word_vectors[word]
        # words not found in embedding index will be all-zeros.
        wv_matrix[i] = embedding_vector
    except:
        pass


# In[23]:


ret_word_index = {t[0]: i+1 for i,t in enumerate(ret_vocab.most_common(num_words-1))}

ret_train_sequences = [[ret_word_index.get(t, 0) for t in sentence]
             for sentence in IMDB_train[:len(IMDB_train)]]

ret_test_sequences = [[ret_word_index.get(t, 0)
                   for t in sentence] for sentence in IMDB_test[:len(IMDB_test)]]

# Pad zeros to match the size of matrix
ret_train_data = pad_sequences(ret_train_sequences, maxlen=seq_length, padding="post", truncating="post")
ret_test_data = pad_sequences(ret_test_sequences, maxlen=seq_length, padding="post", truncating="post")


# In[24]:


# Initialize the matrix with random numbers
ret_wv_matrix = (np.random.rand(num_words, wv_dim) - 0.5) / 5.0
for word, i in ret_word_index.items():
    if i >= num_words:
        continue
    try:
        ret_embedding_vector = ret_word_vectors[word]
        # words not found in embedding index will be all-zeros.
        ret_wv_matrix[i] = ret_embedding_vector
    except:
        pass


# In[30]:


from keras.layers import Reshape, Conv2D, MaxPooling2D, Flatten, Concatenate


# In[57]:


def cnn_1(comment_input):
    wv_layer = Embedding(num_words,
                     wv_dim,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=seq_length,
                     trainable=False)
    
    embedded_sequences = wv_layer(comment_input)

    input_layer=Reshape([2450,100,1])(embedded_sequences)

    conv1=Conv2D(64,kernel_size=(5,100),activation='relu')(input_layer)

    pool1 = MaxPooling2D(pool_size=(2,1))(conv1)

    conv2 = Conv2D(32,kernel_size=(5,1),activation='relu')(pool1)

    pool2 = MaxPooling2D(pool_size=(2,1))(conv2)

    pool2_flat=Flatten()(pool2)

    dropout = Dropout(0.4)(pool2_flat)

    # normalize = BatchNormalization()(dropout)

    logits = Dense(1, activation='sigmoid')(dropout)
    
    return logits


# In[66]:


from sklearn.metrics import f1_score
def train_and_eval(cnn):
    comment_input = Input(shape=(seq_length,), dtype='int64')

    preds= cnn(comment_input)
    model = Model(inputs=[comment_input], outputs=preds)
    model.compile(loss='binary_crossentropy',   #binary_crossentropy
                  optimizer=Adam(lr=0.0005, clipnorm=.25, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
    print(model.summary())

    hist = model.fit(train_data, IMDB_train_y, validation_data=(test_data, IMDB_test_y), epochs=15, batch_size=32)
    
    y_pred_train = model.predict(train_data)
    y_pred_train = [int(item>0.5) for  item in y_pred_train]
    y_pred_test = model.predict(test_data)
    y_pred_test = [int(item>0.5) for  item in y_pred_test]
    f1_train= f1_score(IMDB_train_y, y_pred_train, average='micro')
    f1_test= f1_score(IMDB_test_y, y_pred_test, average='micro')
    print('f1 (train): ', f1_train)
    print('f1 (test): ', f1_test)
    return model


# In[63]:


def cnn_ret(comment_input):
    wv_layer = Embedding(num_words,
                     wv_dim,
                     mask_zero=False,
                     weights=[ret_wv_matrix],
                     input_length=seq_length,
                     trainable=False)
    
    embedded_sequences = wv_layer(comment_input)

    input_layer=Reshape([2450,100,1])(embedded_sequences)

    conv1=Conv2D(64,kernel_size=(5,100),activation='relu')(input_layer)

    pool1 = MaxPooling2D(pool_size=(2,1))(conv1)

    conv2 = Conv2D(32,kernel_size=(5,1),activation='relu')(pool1)

    pool2 = MaxPooling2D(pool_size=(2,1))(conv2)

    pool2_flat=Flatten()(pool2)

    dropout = Dropout(0.4)(pool2_flat)

    # normalize = BatchNormalization()(dropout)

    logits = Dense(1, activation='sigmoid')(dropout)
    
    return logits


# In[64]:


def cnn_2(comment_input):
    wv_layer = Embedding(num_words,
                     wv_dim,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=seq_length,
                     trainable=False)

    ret_wv_layer = Embedding(num_words,
                     wv_dim,
                     mask_zero=False,
                     weights=[ret_wv_matrix],
                     input_length=seq_length,
                     trainable=False)
    # channel1
    embedded_sequences1 = wv_layer(comment_input)

    input_layer_1=Reshape([2450,100,1])(embedded_sequences1)

    conv1_1=Conv2D(64,kernel_size=(5,100),activation='relu')(input_layer_1)

    pool1_1 = MaxPooling2D(pool_size=(2,1))(conv1_1)

    conv2_1 = Conv2D(32,kernel_size=(5,1),activation='relu')(pool1_1)

    pool2_1 = MaxPooling2D(pool_size=(2,1))(conv2_1)

    pool2_flat_1=Flatten()(pool2_1)

    # channel 2
    embedded_sequences2 = ret_wv_layer(comment_input)
    input_layer_2=Reshape([2450,100,1])(embedded_sequences2)

    conv1_2=Conv2D(64,kernel_size=(5,100),activation='relu')(input_layer_2)

    pool1_2 = MaxPooling2D(pool_size=(2,1))(conv1_2)

    conv2_2 = Conv2D(32,kernel_size=(5,1),activation='relu')(pool1_2)

    pool2_2 = MaxPooling2D(pool_size=(2,1))(conv2_2)

    pool2_flat_2=Flatten()(pool2_2)
    
    # merge
    merged = Concatenate(axis=-1)([pool2_flat_1, pool2_flat_2])
    # interpretation
    dropout_2 = Dropout(0.4)(merged)
    
    logits = Dense(1, activation='sigmoid')(dropout_2)
    
    
    
    return logits


# In[46]:


train_and_eval(cnn_1) # kernel size=5, dropout=0.2


# In[55]:


train_and_eval(cnn_1) # kernel size=3


# In[67]:


model_1=train_and_eval(cnn_1) # dropout: 0.4, kernel_size=5


# In[68]:


model_ret=train_and_eval(cnn_ret)


# In[ ]:


model_2=train_and_eval(cnn_2)

