
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import nltk
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import RandomizedSearchCV
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

stemmer = SnowballStemmer("english", ignore_stopwords=True)
lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')
np.set_printoptions(threshold=np.nan)


# In[2]:


# Load data
IMDB_train = pd.read_csv('./IMDB-train.txt', sep='\t', encoding='latin-1', header=None)
IMDB_train_y = IMDB_train[:][1]
IMDB_valid = pd.read_csv('./IMDB-valid.txt', sep='\t', encoding='latin-1', header=None)
IMDB_valid_y = IMDB_valid[:][1]
IMDB_test = pd.read_csv('./IMDB-test.txt', sep='\t', encoding='latin-1', header=None)
IMDB_test_y = IMDB_test[:][1]
stemmer = SnowballStemmer("english", ignore_stopwords=True)

print("Data loaded.")


# In[3]:


frames = [IMDB_train, IMDB_valid]
frames_y = [IMDB_train_y, IMDB_valid_y]
IMDB_train = pd.concat(frames)
IMDB_train_y = pd.concat(frames_y)


# # Preprocessing

# In[4]:


def preprocessing(data):
    new_data = []
    #i = 0
    for sentence in (data[:][0]):
        new_sentence = re.sub('<.*?>', '', sentence) # remove HTML tags
        new_sentence = re.sub(r'[^\w\s]', '', new_sentence) # remove punctuation
        new_sentence = new_sentence.lower() # convert to lower case
        if new_sentence != '':
            new_data.append(new_sentence)
    return new_data


# In[5]:


def rm_numbers(data):
    new_data = []
    #i = 0
    for sentence in (data):
        new_sentence = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", sentence)
        if new_sentence != '':
            new_data.append(new_sentence)
    return new_data


# In[6]:


IMDB_train = preprocessing(IMDB_train)
IMDB_test = preprocessing(IMDB_test)


# In[57]:


IMDB_train_rm_num = rm_numbers(IMDB_train)
IMDB_test_rm_num = rm_numbers(IMDB_test)


# # Bag of n-gram 

# In[7]:


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


# In[10]:



unigram = CountVectorizer(tokenizer=LemmaTokenizer(), analyzer='word', ngram_range=(1, 1), stop_words='english', max_features =30000)
unigram_w_sw = CountVectorizer(tokenizer=LemmaTokenizer(), analyzer='word', ngram_range=(1, 1), stop_words=None, max_features =30000)


# In[11]:


train_unigram = unigram.fit_transform(IMDB_train).toarray()
test_unigram = unigram.transform(IMDB_test).toarray()


# In[12]:


train_unigram_w_sw = unigram_w_sw.fit_transform(IMDB_train).toarray()
test_unigram_w_sw = unigram_w_sw.transform(IMDB_test).toarray()


# In[8]:


unigram_tfid_w_sw = TfidfVectorizer(tokenizer=LemmaTokenizer(),analyzer='word', stop_words=None, ngram_range=(1, 1), max_features =30000)


# In[9]:


train_unigram_tfid_w_sw = unigram_tfid_w_sw.fit_transform(IMDB_train).toarray()
test_unigram_tfid_w_sw = unigram_tfid_w_sw.transform(IMDB_test).toarray()


# # Random & Majority Classifiers as Baseline

# In[50]:


def random_classifier(label):
    pred = np.random.randint(0,2, size=len(label))
    f1= f1_score(label, pred, average='micro')
    return f1
            


# In[51]:


IMDB_train_rc_f1 = random_classifier(IMDB_train_y)
print(IMDB_train_rc_f1)

IMDB_test_rc_f1 = random_classifier(IMDB_test_y)
print(IMDB_test_rc_f1)


# In[52]:


def takeSecond(elem):
    return elem[1]

def majority_classifier_train(label):
    frequency = list(Counter(label).items())
    frequency = sorted(frequency, key=takeSecond, reverse=True)
    majority_class = frequency[0][0]
    pred = np.full(len(label), majority_class)
    f1 = f1_score(label, pred, average='micro')
    return f1, majority_class


def majority_classifier_test(label, majority_class):
    pred = np.full(len(label), majority_class)
    f1 = f1_score(label, pred, average='micro')
    return f1
    


# In[53]:


IMDB_train_mc_f1,IMDB_train_mc_class = majority_classifier_train(IMDB_train_y)
print(IMDB_train_mc_f1,IMDB_train_mc_class)

IMDB_test_mc_f1 = majority_classifier_test(IMDB_test_y, IMDB_train_mc_class)
print(IMDB_test_mc_f1)


# # Naive Bayes Classifier

# In[11]:


def Naive_Bayes_Bernoulli(train_data, train_label, test_data, test_label, cv):

    tuned_parameters = [{'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}]
    clf = BernoulliNB()
    clf = GridSearchCV(clf, tuned_parameters, refit=True, scoring='f1_micro', cv=cv, return_train_score=True)
    clf.fit(train_data, train_label)

    
    train_scores = clf.cv_results_['mean_train_score']
    print('train_scores:',train_scores)
    test_scores = clf.cv_results_['mean_test_score']
    print('valid_scores:',test_scores)
    params = clf.cv_results_['params']
    print('params:', params)
    best_param = clf.best_params_ 
    print('best_param', best_param)
    best_estimator = clf.best_estimator_  
    print('best_estimator', best_estimator)
    best_score = clf.best_score_
    print('best_score', best_score)

    clf = BernoulliNB(alpha = best_param['alpha'])
    clf.fit(train_data, train_label)
    
    y_pred_train = clf.predict(train_data)    
    y_pred_test = clf.predict(test_data)
    f1_train= f1_score(train_label, y_pred_train, average='micro')
    f1_test= f1_score(test_label, y_pred_test, average='micro')
    print('f1 (train): ', f1_train)
    print('f1 (test): ', f1_test)


# In[16]:


Naive_Bayes_Bernoulli(train_unigram,np.asarray(IMDB_train_y),test_unigram,np.asarray(IMDB_test_y), 5)


# In[17]:


Naive_Bayes_Bernoulli(train_unigram_w_sw,IMDB_train_y,test_unigram_w_sw,IMDB_test_y, 5)


# In[15]:


def Naive_Bayes_Gaussian(train_data, train_label, test_data, test_label, cv):

    tuned_parameters = [{'priors': [[0.5,0.5],[0.9,0.1],[0.1,0.9]]}]
    clf = GaussianNB()
    clf = GridSearchCV(clf, tuned_parameters, refit=True, scoring='f1_micro', cv=cv, return_train_score=True)
    clf.fit(train_data, train_label)

    
    train_scores = clf.cv_results_['mean_train_score']
    print('train_scores:',train_scores)
    test_scores = clf.cv_results_['mean_test_score']
    print('valid_scores:',test_scores)
    params = clf.cv_results_['params']
    print('params:', params)
    best_param = clf.best_params_ 
    print('best_param', best_param)
    best_estimator = clf.best_estimator_  
    print('best_estimator', best_estimator)
    best_score = clf.best_score_
    print('best_score', best_score)

    clf = GaussianNB(priors = best_param['priors'])
    clf.fit(train_data, train_label)
    
    y_pred_train = clf.predict(train_data)    
    y_pred_test = clf.predict(test_data)
    f1_train= f1_score(train_label, y_pred_train, average='micro')
    f1_test= f1_score(test_label, y_pred_test, average='micro')
    print('f1 (train): ', f1_train)
    print('f1 (test): ', f1_test)


# In[17]:


Naive_Bayes_Gaussian(train_unigram_tfid_w_sw,IMDB_train_y,test_unigram_tfid_w_sw,IMDB_test_y, 5)


# # Support Vector Machine Classifier

# In[16]:


def SVM(train_data, train_label, test_data, test_label, cv):

    tuned_parameters = [{'C': [0.01, 0.1, 1.0, 2.0, 3.0], 'tol': [0.0001, 0.001, 0.01, 0.1, 1]}]
    
    clf = LinearSVC(dual=False)
    clf = GridSearchCV(clf, tuned_parameters, refit=True, scoring='f1_micro', cv=cv, return_train_score=True)
    clf.fit(train_data, train_label)

    
    train_scores = clf.cv_results_['mean_train_score']
    print('train_scores:',train_scores)
    test_scores = clf.cv_results_['mean_test_score']
    print('valid_scores:',test_scores)
    params = clf.cv_results_['params']
    print('params:', params)
    best_param = clf.best_params_ 
    print('best_param', best_param)
    best_estimator = clf.best_estimator_  
    print('best_estimator', best_estimator)
    best_score = clf.best_score_
    print('best_score', best_score)

    clf = LinearSVC(C=best_param['C'], tol=best_param['tol'], dual=False)
    clf.fit(train_data, train_label)
    
    y_pred_train = clf.predict(train_data)    
    y_pred_test = clf.predict(test_data)
    f1_train= f1_score(train_label, y_pred_train, average='micro')
    f1_test= f1_score(test_label, y_pred_test, average='micro')
    print('f1 (train): ', f1_train)
    print('f1 (test): ', f1_test)


# In[14]:


SVM(train_unigram,IMDB_train_y,test_unigram,IMDB_test_y, 5)


# In[15]:


SVM(train_unigram_w_sw,IMDB_train_y,test_unigram_w_sw,IMDB_test_y, 5)


# In[18]:


SVM(train_unigram_tfid_w_sw,IMDB_train_y,test_unigram_tfid_w_sw,IMDB_test_y, 5)

