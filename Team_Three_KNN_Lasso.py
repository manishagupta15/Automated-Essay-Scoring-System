# -*- coding: utf-8 -*-



## Import all packages and dataset

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re, collections
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('training_set.csv',encoding = "ISO-8859-1")

## use three columns from dataset

allessay = dataset[['essay_set','essay','domain1_score']].copy()

print(allessay)

## Score Scale for each essay set:

get_ipython().run_line_magic('matplotlib', 'inline')
allessay.boxplot(column = 'domain1_score', by = 'essay_set', figsize = (10, 10))

## Get features from all essays
## Acknowledge: defined functions resourced from github website

def sentence_to_wordlist(raw_sentence):
    
    clean_sentence = re.sub("[^a-zA-Z0-9]"," ", raw_sentence)
    tokens = nltk.word_tokenize(clean_sentence)
    
    return tokens

def tokenize(essay):
    stripped_essay = essay.strip()
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)
    
    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(sentence_to_wordlist(raw_sentence))
    
    return tokenized_sentences

def word_count(essay):
    
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)
    
    return len(words)

def avg_word_len(essay):
    
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)
    
    return sum(len(word) for word in words) / len(words)

def sent_count(essay):
    
    sentences = nltk.sent_tokenize(essay)
    
    return len(sentences)

def char_count(essay):
    
    clean_essay = re.sub(r'\s', '', str(essay).lower())
    
    return len(clean_essay)

def count_pos(essay):
    
    tokenized_sentences = tokenize(essay)
    
    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0
    
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)
        
        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]
        
            if pos_tag.startswith('N'): 
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1
            
    return noun_count, adj_count, verb_count, adv_count

def count_lemmas(essay):
    
    tokenized_sentences = tokenize(essay)      
    
    lemmas = []
    wordnet_lemmatizer = WordNetLemmatizer()
    
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence) 
        
        for token_tuple in tagged_tokens:
        
            pos_tag = token_tuple[1]
        
            if pos_tag.startswith('N'): 
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('J'):
                pos = wordnet.ADJ
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            else:
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
    
    lemma_count = len(set(lemmas))
    
    return lemma_count

def count_spell_error(essay):
    
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)

    ## gutenburg file contains a million word can be used in an essay, downloaded from project gutenburg website
    
    data = open('gutenberg.txt').read()
    
    words_ = re.findall('[a-z]+', data.lower())
    
    word_dict = collections.defaultdict(lambda: 0)
                       
    for word in words_:
        word_dict[word] += 1
                       
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)
                        
    mispell_count = 0
    
    words = clean_essay.split()
                        
    for word in words:
        if not word in word_dict:
            mispell_count += 1
    
    return mispell_count

def get_count_vectors(essays):
    
    vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')
    
    count_vectors = vectorizer.fit_transform(essays)
    
    feature_names = vectorizer.get_feature_names()
    
    return feature_names, count_vectors

## Extract ten features from all essays

def extract_features(data):
    
    features = allessay.copy()
    
    features['char_count'] = features['essay'].apply(char_count)
    
    features['word_count'] = features['essay'].apply(word_count)
    
    features['sent_count'] = features['essay'].apply(sent_count)
    
    features['avg_word_len'] = features['essay'].apply(avg_word_len)
    
    features['lemma_count'] = features['essay'].apply(count_lemmas)
    
    features['spell_err_count'] = features['essay'].apply(count_spell_error)
    
    features['noun_count'], features['adj_count'], features['verb_count'], features['adv_count'] = zip(*features['essay'].map(count_pos))
    
    return features

## get features from set1
    
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')  
nltk.download('wordnet')

set1 = extract_features(allessay[allessay['essay_set'] == 1])

print(set1)

## plot a correlation map between each feature and score

get_ipython().run_line_magic('matplotlib', 'inline')
set1.plot.scatter(x = 'char_count', y = 'domain1_score', s=10)
set1.plot.scatter(x = 'word_count', y = 'domain1_score', s=10)
set1.plot.scatter(x = 'sent_count', y = 'domain1_score', s=10)
set1.plot.scatter(x = 'avg_word_len', y = 'domain1_score', s=10)
set1.plot.scatter(x = 'lemma_count', y = 'domain1_score', s=10)
set1.plot.scatter(x = 'spell_err_count', y = 'domain1_score', s=10)
set1.plot.scatter(x = 'noun_count', y = 'domain1_score', s=10)
set1.plot.scatter(x = 'adj_count', y = 'domain1_score', s=10)
set1.plot.scatter(x = 'verb_count', y = 'domain1_score', s=10)
set1.plot.scatter(x = 'adv_count', y = 'domain1_score', s=10)

feature_names_cv, count_vectors = get_count_vectors(allessay[allessay['essay_set'] == 1]['essay'])

X_cv = count_vectors.toarray()

y_cv = allessay[allessay['essay_set'] == 1]['domain1_score'].as_matrix()

## Split dataset and train models

X = np.concatenate((set1.iloc[:, 3:].as_matrix(),X_cv), axis = 1)

y = set1['domain1_score'].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

# define a list of parameters
#param_knn = {'n_neighbors': [5, 10, 15, 20, 25, 30]}
param_knn = {'n_neighbors': range(5,25,3)}

#apply grid search
grid_knn = GridSearchCV(knn, param_knn, cv=5, return_train_score=True)
grid_knn.fit(X_train, y_train)

print('train score: ', grid_knn.score(X_train, y_train))
print('test score: ', grid_knn.score(X_test, y_test))
grid_knn.best_params_

n_neighbors = 23

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
clf_train = KNeighborsClassifier(n_neighbors)
clf_train.fit(X_train, y_train)

print("Test set accuracy: {:.2f}".format(clf_train.score(X_test, y_test)))

alphas = np.array([3, 2, 1, 0.6, 0.3, 0.1])

lasso_regressor = Lasso()

grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

print(grid.best_score_)
print(grid.best_estimator_.alpha)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

print('Variance score: %.2f' % grid.score(X_test, y_test))

## Analyze for set2

set2 = extract_features(allessay[allessay['essay_set'] == 2])

print(set2)

feature_names_cv, count_vectors = get_count_vectors(allessay[allessay['essay_set'] == 2]['essay'])

X_cv2 = count_vectors.toarray()

y_cv2 = allessay[allessay['essay_set'] == 2]['domain1_score'].as_matrix()

X2 = np.concatenate((set2.iloc[:, 3:].as_matrix(),X_cv2), axis = 1)

y2 = set2['domain1_score'].as_matrix()

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.3)

knn = KNeighborsClassifier()

# define a list of parameters
#param_knn = {'n_neighbors': [5, 10, 15, 20, 25, 30]}
param_knn = {'n_neighbors': range(5,25,3)}

#apply grid search
grid_knn = GridSearchCV(knn, param_knn, cv=5, return_train_score=True)
grid_knn.fit(X_train2, y_train2)

print('train score: ', grid_knn.score(X_train2, y_train2))
print('test score: ', grid_knn.score(X_test2, y_test2))
grid_knn.best_params_

n_neighbors = 23

clf_train = KNeighborsClassifier(n_neighbors)
clf_train.fit(X_train2, y_train2)

print("Test set accuracy: {:.2f}".format(clf_train.score(X_test2, y_test2)))

alphas = np.array([3, 2, 1, 0.6, 0.3, 0.1])

lasso_regressor = Lasso()

grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))
grid.fit(X_train2, y_train2)

y_pred2 = grid.predict(X_test2)

print(grid.best_score_)
print(grid.best_estimator_.alpha)

print("Mean squared error: %.2f" % mean_squared_error(y_test2, y_pred2))

print('Variance score: %.2f' % grid.score(X_test2, y_test2))

## Analyze for set3

set3 = extract_features(allessay[allessay['essay_set'] == 3])

print(set3)

feature_names_cv, count_vectors = get_count_vectors(allessay[allessay['essay_set'] == 3]['essay'])

X_cv3 = count_vectors.toarray()

y_cv3 = allessay[allessay['essay_set'] == 3]['domain1_score'].as_matrix()

X3 = np.concatenate((set3.iloc[:, 3:].as_matrix(),X_cv3), axis = 1)

y3 = set3['domain1_score'].as_matrix()

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size = 0.3)

knn = KNeighborsClassifier()

# define a list of parameters
#param_knn = {'n_neighbors': [5, 10, 15, 20, 25, 30]}
param_knn = {'n_neighbors': range(5,25,3)}

#apply grid search
grid_knn = GridSearchCV(knn, param_knn, cv=5, return_train_score=True)
grid_knn.fit(X_train3, y_train3)

print('train score: ', grid_knn.score(X_train3, y_train3))
print('test score: ', grid_knn.score(X_test3, y_test3))
grid_knn.best_params_

n_neighbors = 17

clf_train = KNeighborsClassifier(n_neighbors)
clf_train.fit(X_train3, y_train3)

print("Test set accuracy: {:.2f}".format(clf_train.score(X_test3, y_test3)))

alphas = np.array([3, 2, 1, 0.6, 0.3, 0.1])

lasso_regressor = Lasso()

grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))
grid.fit(X_train3, y_train3)

y_pred3 = grid.predict(X_test3)

print(grid.best_score_)
print(grid.best_estimator_.alpha)

print("Mean squared error: %.2f" % mean_squared_error(y_test3, y_pred3))

print('Variance score: %.2f' % grid.score(X_test3, y_test3))

## Analyze for set4

set4 = extract_features(allessay[allessay['essay_set'] == 4])

print(set4)

feature_names_cv, count_vectors = get_count_vectors(allessay[allessay['essay_set'] == 4]['essay'])

X_cv4 = count_vectors.toarray()

y_cv4 = allessay[allessay['essay_set'] == 4]['domain1_score'].as_matrix()

X4 = np.concatenate((set4.iloc[:, 3:].as_matrix(),X_cv4), axis = 1)

y4 = set4['domain1_score'].as_matrix()

X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size = 0.3)

knn = KNeighborsClassifier()

# define a list of parameters
#param_knn = {'n_neighbors': [5, 10, 15, 20, 25, 30]}
param_knn = {'n_neighbors': range(5,25,3)}

#apply grid search
grid_knn = GridSearchCV(knn, param_knn, cv=5, return_train_score=True)
grid_knn.fit(X_train4, y_train4)

print('train score: ', grid_knn.score(X_train4, y_train4))
print('test score: ', grid_knn.score(X_test4, y_test4))
grid_knn.best_params_

n_neighbors = 23

clf_train = KNeighborsClassifier(n_neighbors)
clf_train.fit(X_train4, y_train4)

print("Test set accuracy: {:.2f}".format(clf_train.score(X_test4, y_test4)))

alphas = np.array([3, 2, 1, 0.6, 0.3, 0.1])

lasso_regressor = Lasso()

grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))
grid.fit(X_train4, y_train4)

y_pred4 = grid.predict(X_test4)

print(grid.best_score_)
print(grid.best_estimator_.alpha)

print("Mean squared error: %.2f" % mean_squared_error(y_test4, y_pred4))

print('Variance score: %.2f' % grid.score(X_test4, y_test4))

## Analyze for set5

set5 = extract_features(allessay[allessay['essay_set'] == 5])

print(set5)

feature_names_cv, count_vectors = get_count_vectors(allessay[allessay['essay_set'] == 5]['essay'])

X_cv5 = count_vectors.toarray()

y_cv5 = allessay[allessay['essay_set'] == 5]['domain1_score'].as_matrix()

X5 = np.concatenate((set5.iloc[:, 3:].as_matrix(),X_cv5), axis = 1)

y5 = set5['domain1_score'].as_matrix()

X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y5, test_size = 0.3)

knn = KNeighborsClassifier()

# define a list of parameters
#param_knn = {'n_neighbors': [5, 10, 15, 20, 25, 30]}
param_knn = {'n_neighbors': range(5,25,3)}

#apply grid search
grid_knn = GridSearchCV(knn, param_knn, cv=5, return_train_score=True)
grid_knn.fit(X_train5, y_train5)

print('train score: ', grid_knn.score(X_train5, y_train5))
print('test score: ', grid_knn.score(X_test5, y_test5))
grid_knn.best_params_

n_neighbors = 23

clf_train = KNeighborsClassifier(n_neighbors)
clf_train.fit(X_train5, y_train5)

print("Test set accuracy: {:.2f}".format(clf_train.score(X_test5, y_test5)))

alphas = np.array([3, 2, 1, 0.6, 0.3, 0.1])

lasso_regressor = Lasso()

grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))
grid.fit(X_train5, y_train5)

y_pred5 = grid.predict(X_test5)

print(grid.best_score_)
print(grid.best_estimator_.alpha)

print("Mean squared error: %.2f" % mean_squared_error(y_test5, y_pred5))

print('Variance score: %.2f' % grid.score(X_test5, y_test5))

## Analyze for set6:

set6 = extract_features(allessay[allessay['essay_set'] == 6])

print(set6)

feature_names_cv, count_vectors = get_count_vectors(allessay[allessay['essay_set'] == 6]['essay'])

X_cv6 = count_vectors.toarray()

y_cv6 = allessay[allessay['essay_set'] == 6]['domain1_score'].as_matrix()

X6 = np.concatenate((set6.iloc[:, 3:].as_matrix(),X_cv6), axis = 1)

y6 = set6['domain1_score'].as_matrix()

X_train6, X_test6, y_train6, y_test6 = train_test_split(X6, y6, test_size = 0.3)

knn = KNeighborsClassifier()

# define a list of parameters
#param_knn = {'n_neighbors': [5, 10, 15, 20, 25, 30]}
param_knn = {'n_neighbors': range(5,25,3)}

#apply grid search
grid_knn = GridSearchCV(knn, param_knn, cv=5, return_train_score=True)
grid_knn.fit(X_train6, y_train6)

print('train score: ', grid_knn.score(X_train6, y_train6))
print('test score: ', grid_knn.score(X_test6, y_test6))
grid_knn.best_params_

n_neighbors = 23

clf_train = KNeighborsClassifier(n_neighbors)
clf_train.fit(X_train6, y_train6)

print("Test set accuracy: {:.2f}".format(clf_train.score(X_test6, y_test6)))

alphas = np.array([3, 2, 1, 0.6, 0.3, 0.1])

lasso_regressor = Lasso()

grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))
grid.fit(X_train6, y_train6)

y_pred6 = grid.predict(X_test6)

print(grid.best_score_)
print(grid.best_estimator_.alpha)

print("Mean squared error: %.2f" % mean_squared_error(y_test6, y_pred6))

print('Variance score: %.2f' % grid.score(X_test6, y_test6))

## Analyze for set7

set7 = extract_features(allessay[allessay['essay_set'] == 7])

print(set7)

feature_names_cv, count_vectors = get_count_vectors(allessay[allessay['essay_set'] == 7]['essay'])

X_cv7 = count_vectors.toarray()

y_cv7 = allessay[allessay['essay_set'] == 7]['domain1_score'].as_matrix()

X7 = np.concatenate((set7.iloc[:, 3:].as_matrix(),X_cv7), axis = 1)

y7 = set7['domain1_score'].as_matrix()

X_train7, X_test7, y_train7, y_test7 = train_test_split(X7, y7, test_size = 0.3)

knn = KNeighborsClassifier()

# define a list of parameters
#param_knn = {'n_neighbors': [5, 10, 15, 20, 25, 30]}
param_knn = {'n_neighbors': range(5,25,3)}

#apply grid search
grid_knn = GridSearchCV(knn, param_knn, cv=5, return_train_score=True)
grid_knn.fit(X_train7, y_train7)

print('train score: ', grid_knn.score(X_train7, y_train7))
print('test score: ', grid_knn.score(X_test7, y_test7))
grid_knn.best_params_

n_neighbors = 11

clf_train = KNeighborsClassifier(n_neighbors)
clf_train.fit(X_train7, y_train7)

print("Test set accuracy: {:.2f}".format(clf_train.score(X_test7, y_test7)))

alphas = np.array([3, 2, 1, 0.6, 0.3, 0.1])

lasso_regressor = Lasso()

grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))
grid.fit(X_train7, y_train7)

y_pred7 = grid.predict(X_test7)

print(grid.best_score_)
print(grid.best_estimator_.alpha)

print("Mean squared error: %.2f" % mean_squared_error(y_test7, y_pred7))

print('Variance score: %.2f' % grid.score(X_test7, y_test7))

## Analyze for set8

set8 = extract_features(allessay[allessay['essay_set'] == 8])

print(set8)

feature_names_cv, count_vectors = get_count_vectors(allessay[allessay['essay_set'] == 8]['essay'])

X_cv8 = count_vectors.toarray()

y_cv8 = allessay[allessay['essay_set'] == 8]['domain1_score'].as_matrix()

X8 = np.concatenate((set8.iloc[:, 3:].as_matrix(),X_cv8), axis = 1)

y8 = set8['domain1_score'].as_matrix()

X_train8, X_test8, y_train8, y_test8 = train_test_split(X8, y8, test_size = 0.3)

knn = KNeighborsClassifier()

# define a list of parameters
#param_knn = {'n_neighbors': [5, 10, 15, 20, 25, 30]}
param_knn = {'n_neighbors': range(5,25,3)}

#apply grid search
grid_knn = GridSearchCV(knn, param_knn, cv=5, return_train_score=True)
grid_knn.fit(X_train8, y_train8)

print('train score: ', grid_knn.score(X_train8, y_train8))
print('test score: ', grid_knn.score(X_test8, y_test8))
grid_knn.best_params_

n_neighbors = 23

clf_train = KNeighborsClassifier(n_neighbors)
clf_train.fit(X_train8, y_train8)

print("Test set accuracy: {:.2f}".format(clf_train.score(X_test8, y_test8)))

alphas = np.array([3, 2, 1, 0.6, 0.3, 0.1])

lasso_regressor = Lasso()

grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))
grid.fit(X_train8, y_train8)

y_pred8 = grid.predict(X_test8)

print(grid.best_score_)
print(grid.best_estimator_.alpha)

print("Mean squared error: %.2f" % mean_squared_error(y_test8, y_pred8))

print('Variance score: %.2f' % grid.score(X_test8, y_test8))

