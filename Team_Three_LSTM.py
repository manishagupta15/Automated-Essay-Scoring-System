# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 14:37:32 2019

@author: Manisha Gupta
"""


import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import os
import pandas as pd

from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score

import tensorflow as tf
import  keras.layers  as  klayers 
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Input, Embedding, GlobalAveragePooling1D, Concatenate, Activation, Lambda, BatchNormalization, Convolution1D, Dropout
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import initializers
from scipy import stats
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from keras.initializers import Constant
import numpy as np
import csv
import textstat
# Constants
DATASET_DIR = r'C:\Users\AbhiMan\Documents\UTDALLAS\MS SYLBUS UTD\BUAN 6341\project\data'
GLOVE_DIR = './glove.6B/'
SAVE_DIR = './'

EMBEDDING_DIM = 300
MAX_NB_WORDS = 4000
MAX_SEQUENCE_LENGTH = 500
VALIDATION_SPLIT= 0.20
DELTA = 20



#def getembedded():
#    embeddings_index = {}
#    with open(r'C:\Users\AbhiMan\Documents\UTDALLAS\MS SYLBUS UTD\BUAN 6341\project\data\glove.6B\glove.6B.300d.txt', encoding="utf-8") as f:
#        for line in f:
#            values = line.split()
#            word = values[0]
#            coefs = np.asarray(values[1:], dtype='float32')
#            embeddings_index[word] = coefs
#    return embeddings_index


def feature_getter(text):
    try:
        text=text.decode('utf-8')
    except:
        pass
    text1=re.sub(r'[^\x00-\x7F]+',' ', text)
    ##text1=re.sub('\n','. ', text)
    text=text1
    features=[]
    tokens=[]
    sentences = nltk.sent_tokenize(text)
    [tokens.extend(nltk.word_tokenize(sentence)) for sentence in sentences]
    
    syllable_count = textstat.syllable_count(text, lang='en_US')
    word_count = textstat.lexicon_count(text, removepunct=True)

    flesch = textstat.flesch_reading_ease(text)
    readability = textstat.automated_readability_index(text)

    features.append(len(sentences)) #num_sentences
    features.append(syllable_count) #num_sentences
    features.append(word_count) #num_sentences
    features.append(flesch) #num_sentences
    features.append(readability) #num_sentences       
    return features

def get_documents():
    
    datals = []
    
    hp_score = pd.read_csv(os.path.join(DATASET_DIR, 'training_set_rel3.tsv'), sep='\t', encoding='ISO-8859-1')


    
    for indx in hp_score.index: 
        idsteem = hp_score.loc[indx, "essay_id"]
        score = hp_score.loc[indx, "domain1_score"]
        essaytype = hp_score.loc[indx, "essay_set"]
        essay = hp_score.loc[indx, "essay"]
        score1 = 0
   
        if essaytype==1:
            score1 = (score-2)/10 
        if essaytype==2:
            score1 = (score-1)/5      
        if essaytype==3:
            score1 = (score-0)/3 
        if essaytype==4:
            score1 = (score-0)/3      
        if essaytype==5:
            score1 = (score-2)/4 
        if essaytype==6:
            score1 = (score-2)/4   
        if essaytype==7:
            score1 = (score-0)/30
        if essaytype==8:
            score1 = (score-0)/60
        datals.append({"id":idsteem,  "score": score1, "essay": essay})
        
       
    X = pd.DataFrame(datals)
    return X

def get_model():
    """Define the model."""
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.3, input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model

def essay_to_wordlist(essay_v, remove_stopwords):
    """Remove the tagged labels and word tokenize the sentence."""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def essay_to_sentences(essay_v, remove_stopwords):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model,index2word_set, num_features):
    """Make Feature Vector from the words list of an Essay."""
    featureVec = np.zeros((num_features,),dtype="float32")
    num_words = 0.

    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec,model[index2word_set[word]])        
    featureVec = np.divide(featureVec,num_words)
    return featureVec

def getAvgFeatureVecs(essays, model, index2word_set, num_features):
    """Main function to generate the word vectors for word2vec model."""
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model,index2word_set, num_features)
        counter = counter + 1
    return essayFeatureVecs


def convertword2vec(dim=300):
    word2vec = []
    word_idx = {}
    word2vec.append([0]*dim)
    count = 1
    with open(r'C:\Users\AbhiMan\Documents\UTDALLAS\MS SYLBUS UTD\BUAN 6341\project\data\glove.6B\glove.6B.300d.txt', encoding="utf-8") as f:
        for line in f:
            l = line.split()
            word = l[0]
            vector = list(map(float, l[1:]))
            word_idx[word] = count
            word2vec.append(vector)
            count += 1
    return word_idx, word2vec

if __name__ == '__main__':
    

    dirpathname= r"C:\Users\AbhiMan\Documents\UTDALLAS\MS SYLBUS UTD\BUAN 6341\project\model"
    os.chdir(dirpathname)
    
    word_idx, word2vec = convertword2vec()
    X = get_documents()
    X = X.dropna(axis=1)

    y = X['score']
    
   # print(len(X))
    
    features = []
    
    for indx in X.index:
        features.append(feature_getter(X.loc[indx, "essay"]))
   # print("Features: ",features)   
    pd.DataFrame(features).to_csv("textssummary.csv", index =False)
    
    xfeature = pd.DataFrame(features)
    #X.head()
    X.to_csv("Scoretextssummary.csv", index =False)
 
    X = X.drop_duplicates(subset = ["id"], keep='first')
    X.index = X.id
    

    X = X.dropna(axis=1)
    y = X['score']
    X.index = range(len(X))
    
    #print(len(X))
  
    indices = np.array(X.index.values)
    np.random.shuffle(indices)
    VALIDATION_SPLIT = 0.20
    num_validation_samples = int(VALIDATION_SPLIT * len(indices))
    
    train = indices[:-num_validation_samples]        
    test =  indices[-num_validation_samples:]
    
    

    X_test, X_train, y_test, y_train = X.iloc[test], X.iloc[train], y.iloc[test], y.iloc[train]
        
    train_essays = X_train['essay']
    test_essays = X_test['essay']
        

    num_features = 300 
    min_word_count = 1
    num_workers = 4
    context = 10
    downsampling = 1e-3
    clean_train_essays = []
    additional_features = []
    
    count = 0
    # Generate training and testing data word vectors.
    for essay_v in train_essays:
        clean_train_essays.append(essay_to_wordlist(essay_v, remove_stopwords=True))
        #additional_features.append(feature_getter(essay_v))
        additional_features.append(0)
        
    trainadditional_features=np.asarray(additional_features)
    
    trainDataVecs = getAvgFeatureVecs(clean_train_essays, word2vec,word_idx, num_features)
    trainIndex = (~np.isnan(trainDataVecs).any(axis=1))
    
        
        
    trainDataVecstrain = trainDataVecs[trainIndex]
    trainAddFeature  = trainadditional_features[trainIndex]
    y_traintrain = y_train[trainIndex]
    
    clean_test_essays = []
    additional_features = []
    for essay_v in test_essays:
        clean_test_essays.append(essay_to_wordlist( essay_v, remove_stopwords=True ))
        #additional_features.append(feature_getter(essay_v))
        additional_features.append(0)
        
    testadditional_features=np.asarray(additional_features)   
    testDataVecs = getAvgFeatureVecs( clean_test_essays, word2vec, word_idx, num_features)
    
    testIndex = (~np.isnan(testDataVecs).any(axis=1))
    testDataVecstest = testDataVecs[testIndex]
    testAddFeature  = testadditional_features[testIndex]
    y_testtest = y_test[testIndex]
    
    trainDataVecstrain = np.array(trainDataVecstrain)
    testDataVecstest = np.array(testDataVecstest)

        
    trainDataVecstrain = np.reshape(trainDataVecstrain, (trainDataVecstrain.shape[0], 1, trainDataVecstrain.shape[1]))
    testDataVecstest = np.reshape(testDataVecstest, (testDataVecstest.shape[0], 1, testDataVecstest.shape[1]))
    
    lstm_model = get_model()
    lstm_model.fit(trainDataVecstrain, y_traintrain, batch_size=64, epochs=50)
    #lstm_model.load_weights('./model_weights/final_lstm.h5')
    y_pred = lstm_model.predict(testDataVecstest)
    
    
    comparesdf = []
    for j in range(len(y_pred)):
        comparesdf.append({"y": y_testtest.values[j], "yhat": y_pred[j][0]})
        #comparesdf.head(5)
        pd.DataFrame(comparesdf).to_csv("pred_pred_yhat.csv", index = False)
        
        # Save any one of the 8 models.
    if count>=0:
         lstm_model.save(r'C:\Users\AbhiMan\Documents\UTDALLAS\MS SYLBUS UTD\BUAN 6341\project\final_LSTM_5000_sample.h5')
        
        # Round y_pred to the nearest integer.
    y_pred2 = np.around(y_pred*100)
   
    y_test = np.around(y_testtest*100)
        

    # Evaluate the model on the evaluation metric. "Quadratic mean averaged Kappa"
    result = cohen_kappa_score(y_test.values,y_pred2,weights='quadratic')
    print("Kappa Score: {}".format(result))
    #results.append(result)
    
        
