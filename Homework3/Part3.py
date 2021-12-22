"""
@author: Gordon Ng
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import gensim
from gensim.models import Word2Vec
import re
import nltk
nltk.download("wordnet")
nltk.download("punkt")
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
from nltk.corpus import stopwords
stopwords_set = set(stopwords.words("English"))
from sklearn.decomposition import PCA

def Problem3_1():
    read_plays_csv = pd.read_csv("will_play_text.csv", sep=";", header=None)
    
    vocabularyfile = pd.read_csv("vocab.txt", sep=";", header=None)
    play_names = pd.read_csv("play_names.txt", sep=";", header=None)
    vocabularyfile.split("\n")[:-1]
    play_names.split("\n")[:-1]
    
    Column1 = read_plays_csv.groupby(1)[1].apply(lambda x: list(x))  # name of play
    #Column2 = read_plays_csv.groupby(1)[2].aggregation()   # who said what
    plays = read_plays_csv.groupby(1)[5].apply(lambda x: list(x))    # text
    
    punc = ['!','(',')','-','[',']','{','}',';', ':', '!', "*"]
    
    total = ["".join(i) for i in plays]
    # new = []
    # for i in total:
    #     new.append(re.findall(r"[\w']+|[.,!?;]", i))
    # total2 = ["".join(i) for i in new if not i in punc]
    # print(total)
    book = CountVectorizer(vocabulary=vocabularyfile)
    X_train = book.fit_transform(total)
    axis = np.asarray(X_train)
    termdocumentmatrix = pd.DataFrame(axis.transpose(), index = vocabularyfile, columns=play_names)
    
    # book.fit_transform(total)
    
    #pd.DataFrame(array, index = , columns=)
    #Array = book.fit_transform(Total)
#Problem3_1()
def Problem3_2():
    read_plays_csv = pd.read_csv("will_play_text.csv", sep=";", header=None)
    
    vocabularyfile = pd.read_csv("vocab.txt", sep=";", header=None)
    play_names = pd.read_csv("play_names.txt", sep=";", header=None)
    vocabularyfile.split("\n")[:-1]
    play_names.split("\n")[:-1]
    
    Column1 = read_plays_csv.groupby(1)[1].apply(lambda x: list(x))  # name of play
    #Column2 = read_plays_csv.groupby(1)[2].aggregation()   # who said what
    plays = read_plays_csv.groupby(1)[5].apply(lambda x: list(x))    # text
    
    punc = ['!','(',')','-','[',']','{','}',';', ':', '!', "*"]
    
    total = ["".join(i) for i in plays]
    book = CountVectorizer(vocabulary=vocabularyfile)
    X_train = book.fit_transform(total)
    axis = np.asarray(X_train)
    termdocumentmatrix = pd.DataFrame(axis.transpose(), index = vocabularyfile, columns=play_names)
    
    PCAresult = PCA(n_components=2)
    functionresult = PCAresult.fit_transform(termdocumentmatrix.transpose())
    pyplot.scatter(functionresult[:, 0], functionresult[:, 1])
    for i, word in enumerate(termdocumentmatrix):
    	pyplot.annotate(word, xy = (functionresult[i, 0], functionresult[i, 1]))
    pyplot.show()
Problem3_2()
def Problem3_3():
    read_plays_csv = pd.read_csv("will_play_text.csv", sep=";", header=None)
    vocabularyfile = pd.read_csv("vocab.txt", sep=";", header=None)
    play_names = pd.read_csv("play_names.txt", sep=";", header=None)
    vocabularyfile.split("\n")[:-1]
    play_names.split("\n")[:-1]
    Column1 = read_plays_csv.groupby(1)[1].apply(lambda x: list(x))  # name of play
    #Column2 = read_plays_csv.groupby(1)[2].aggregation()   # who said what
    plays = read_plays_csv.groupby(1)[5].apply(lambda x: list(x))    # text
    punc = ['!','(',')','-','[',']','{','}',';', ':', '!', "*"]
    total = ["".join(i) for i in plays]
    book = CountVectorizer(vocabulary=vocabularyfile)
    X_train = book.fit_transform(total)
    axis = np.asarray(X_train)
    termdocumentmatrix = pd.DataFrame(axis.transpose(), index = vocabularyfile, columns=play_names)
    PCAresult = PCA(n_components=2)
    functionresult = PCAresult.fit_transform(termdocumentmatrix.transpose())
    complete = TfidfVectorizer(vocabulary=vocabularyfile)
    Y_train = complete.fit_transform(total)
    newArray = Y_train.toarray()
    completematrix = pd.DataFrame(newArray.transpose(), index = vocabularyfile, columns=play_names)
def Problem3_4():
    read_plays_csv = pd.read_csv("will_play_text.csv", sep=";", header=None)
    vocabularyfile = pd.read_csv("vocab.txt", sep=";", header=None)
    play_names = pd.read_csv("play_names.txt", sep=";", header=None)
    vocabularyfile.split("\n")[:-1]
    play_names.split("\n")[:-1]
    Column1 = read_plays_csv.groupby(1)[1].apply(lambda x: list(x))  # name of play
    #Column2 = read_plays_csv.groupby(1)[2].aggregation()   # who said what
    plays = read_plays_csv.groupby(1)[5].apply(lambda x: list(x))    # text
    punc = ['!','(',')','-','[',']','{','}',';', ':', '!', "*"]
    total = ["".join(i) for i in plays]
    book = CountVectorizer(vocabulary=vocabularyfile)
    X_train = book.fit_transform(total)
    axis = np.asarray(X_train)
    termdocumentmatrix = pd.DataFrame(axis.transpose(), index = vocabularyfile, columns=play_names)
    PCAresult = PCA(n_components=2)
    functionresult = PCAresult.fit_transform(termdocumentmatrix.transpose())
    complete = TfidfVectorizer(vocabulary=vocabularyfile)
    Y_train = complete.fit_transform(total)
    newArray = Y_train.toarray()
    completematrix = pd.DataFrame(newArray.transpose(), index = vocabularyfile, columns=play_names)
    matrixmodel= CountVectorizer(vocabulary=vocabularyfile) 
    X = matrixmodel.fit_transform(total).toarray()
    multiplication = X.transpose().dot(X)
    answer = pd.DataFrame(multiplication, index = vocabularyfile, columns = vocabularyfile)


