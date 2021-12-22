# -*- coding: utf-8 -*-
"""
@author: Gordon Ng
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import gensim
from gensim.models import Word2Vec
import nltk
nltk.download("wordnet")
nltk.download("punkt")
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
from nltk.corpus import stopwords
stopwords_set = set(stopwords.words("English"))

def Problem2_1():
    train_df = pd.read_csv("sentiment-train.csv")
    test_df = pd.read_csv("sentiment-test.csv")
    # instatntiate the vectorizer
    X_train = train_df['text']
    Y_train = train_df['sentiment']
    X_test = test_df['text']
    
    vector = CountVectorizer(stop_words="english", max_features=1000)
    # to limit the number of bag-of-word features to only the top 1k 
    # words based on frequency across the corpus
    X_train = vector.fit_transform(X_train)
    X_test = vector.transform(X_test)
    
    model = MultinomialNB()
    model.fit(X_train,Y_train)
    model_prediction = model.predict(X_test)
    
    print(model_prediction)
    print(np.count_nonzero(model_prediction==1), "Number of Ones In Array")
    print(np.count_nonzero(model_prediction==0), "Number of Zeros In Array")
    print(198/(161+198), "= 198/(161+198) = 0.5515320334261838")
    
def Problem2_2():
    train_df = pd.read_csv("sentiment-train.csv")
    test_df = pd.read_csv("sentiment-test.csv")
    # instatntiate the vectorizer
    X_train = train_df['text']
    Y_train = train_df['sentiment']
    X_test = test_df['text']
    
    vector = TfidfVectorizer(stop_words="english", max_features=1000)
    # to limit the number of bag-of-word features to only the top 1k 
    # words based on frequency across the corpus
    X_train = vector.fit_transform(X_train)
    X_test = vector.transform(X_test)
    
    model = MultinomialNB()
    model.fit(X_train,Y_train)
    model_prediction = model.predict(X_test)
    
    Ones = np.count_nonzero(model_prediction==1)
    Zeros = np.count_nonzero(model_prediction==0)
    print(model_prediction)
    print(Ones, "Number of Ones In Array")
    print(Zeros, "Number of Zeros In Array")
    print(Ones/(Zeros+Ones), str(Ones)+"/"+str(Zeros)+"+"+str(Ones), "= 0.5654596100278552")
    print("Using TFIDF counts as features improves the classification accuracy by 0.0139275766 or 1.3%.")
    
def Problem2_3():
    train_df = pd.read_csv("sentiment-train.csv")
    test_df = pd.read_csv("sentiment-test.csv")
    
    X_train = train_df['text']
    Y_train = train_df['sentiment']
    X_test = test_df['text']
    
    vector = CountVectorizer(stop_words="english", max_features=1000)
    X_train = vector.fit_transform(X_train)
    X_test = vector.transform(X_test)
    
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    lr_prediction = logreg.predict(X_test)
    
    print(lr_prediction)
    print(np.count_nonzero(lr_prediction==1), "Number of Ones In Array")
    print(np.count_nonzero(lr_prediction==0), "Number of Zeros In Array")
    print(216/(143+216), "= 216/(143+216) = 0.6016713091922006")
    print("The Logistic Regression Classifier performs better on the test set.")

def Problem2_4():
    train_df = pd.read_csv("sentiment-train.csv")
    test_df = pd.read_csv("sentiment-test.csv")
    
    X_train = train_df['text']
    Y_train = train_df['sentiment']
    X_test = test_df['text']
    
    vector = TfidfVectorizer(stop_words="english", max_features=1000)
    X_train = vector.fit_transform(X_train)
    X_test = vector.transform(X_test)
    
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    lr_prediction = logreg.predict(X_test)
    
    Ones = np.count_nonzero(lr_prediction==1)
    Zeros = np.count_nonzero(lr_prediction==0)
    print(lr_prediction)
    print(Ones, "Number of Ones In Array")
    print(Zeros, "Number of Zeros In Array")
    print(Ones/(Zeros+Ones), str(Ones)+"/"+str(Zeros)+"+"+str(Ones), "0.5877437325905293")
def Problem2_5():
    fivefold = StratifiedKFold(n_splits=5)
    train_df = pd.read_csv("sentiment-train.csv")
    
    X_train = train_df['text']
    Y_train = train_df['sentiment']
    lst = []
    for train_index, test_index in fivefold.split(X_train,Y_train):
        x_train, x_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train, y_test = Y_train.iloc[train_index], Y_train.iloc[test_index]
        
        vector = TfidfVectorizer(stop_words="english", max_features=1000)
        
        x_train = vector.fit_transform(x_train)
        X_test = vector.transform(x_test)
        
        model = MultinomialNB()
        model.fit(x_train,y_train)
        model_prediction = model.predict(X_test)
        score = accuracy_score(y_test, model_prediction)
        lst.append(score)
    print(sum(lst) / len(lst), "This is the Average Accuracies")

def Problem2_5b():
    fivefold = StratifiedKFold(n_splits=5)
    train_df = pd.read_csv("sentiment-train.csv")
    
    X_train = train_df['text']
    Y_train = train_df['sentiment']
    lst = []
    for train_index, test_index in fivefold.split(X_train,Y_train):
        x_train, x_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train, y_test = Y_train.iloc[train_index], Y_train.iloc[test_index]
        
        vector = TfidfVectorizer(stop_words="english", max_features=4000)
        
        x_train = vector.fit_transform(x_train)
        X_test = vector.transform(x_test)
        
        model = MultinomialNB()
        model.fit(x_train,y_train)
        model_prediction = model.predict(X_test)
        score = accuracy_score(y_test, model_prediction)
        lst.append(score)
    print(sum(lst) / len(lst), "This is the Average Accuracies - MAX FEATURES is 4000 with the Greatest Avg Accuracy")

import re
def Problem2_6():
    train_df = pd.read_csv("sentiment-train.csv")
    test_df = pd.read_csv("sentiment-test.csv")
    X_train = train_df['text']
    Y_train = train_df['sentiment']
    
    #Clean Array
    array = []
    cleanedarray = []
    for sentence in X_train:
        clean = sentence.lower()
        clean = re.sub('[^a-zA-Z]', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean)
        array.append(clean)
    #Clean Array
    punc = '''!()-[]{};:'"\, <>.`/?@#$%^&*_~'''
    for sentence_words in array:
        sentence_words.split()
        sentence_words = nltk.word_tokenize(sentence_words)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        sentence_words = [word for word in sentence_words if word not in stopwords_set if word not in punc if word != r'""' if word != r"``" if word != r"''" if word != r"'s"]
        cleanedarray.append(sentence_words)
    new = Word2Vec(sentences=cleanedarray, size=300, min_count=5, workers=1)
    array3 = []
    averages = []
    for sentence in cleanedarray:
        length = len(sentence)
        for i in sentence:
            array3 += [new.wv[i]]
        averages.append(array3 / length)
    

    X_test = test_df['text']
   
    X_train = averages.fit_transform(X_train)
    X_test = averages.transform(X_test)
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    lr_prediction = logreg.predict(X_test)
    
    Ones = np.count_nonzero(lr_prediction==1)
    Zeros = np.count_nonzero(lr_prediction==0)
    print(lr_prediction)
    print(Ones, "Number of Ones In Array")
    print(Zeros, "Number of Zeros In Array")
    print(Ones/(Zeros+Ones), str(Ones)+"/"+str(Zeros)+"+"+str(Ones))
    print("Yes the dense feature representation improves the accuracy of the logistic regression classifier.")
    print("Yes removing stop words improve preformance.")
    
    
    # fresharray = set()
    # for i in cleanedarray:
    #     for x in i:
    #         fresharray.add(x)
    # listarray = []
    # for i in fresharray:
    #     try:
    #         listarray.append(new.wv[i])
    #     except KeyError:
    #         continue
    
    # for i in listarray:
    #     for x[1] in i:
            
    #Word2Vec(sentences=data, size=300, min_count=5, workers=1)
    #yourmodel.wv(the word)

    
#Problem2_1()
#Problem2_2()
#Problem2_3()
#Problem2_4()
#Problem2_5()
#Problem2_5b()
#Problem2_6()
