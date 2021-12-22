# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:55:48 2021

@author: wilso
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from collections import Counter
import nltk
from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models import Word2Vec
import gensim
from sklearn.metrics.pairwise import cosine_similarity
#3.1
play_names = open("play_names.txt","r").read().split("\n")[:-1]
vocab = open("vocab.txt", "r").read().split("\n")[:-1]


review_data = pd.read_csv("will_play_text.csv", sep=";", header=None)



seperatedplays = review_data.groupby(1)[5].apply(lambda x: list(np.unique(x)))


cv = CountVectorizer(vocabulary=vocab)



all_text = ["".join(text) for text in seperatedplays]
X = cv.fit_transform(all_text)
array = X.toarray()



cvmatrix = pd.DataFrame(array.transpose(), index = vocab, columns=play_names)


#3.2
cvpca = PCA(n_components=2)
cvresult = cvpca.fit_transform(cvmatrix.transpose())
pyplot.scatter(cvresult[:, 0], cvresult[:, 1])
for i, word in enumerate(cvmatrix):
	pyplot.annotate(word, xy=(cvresult[i, 0], cvresult[i, 1]))
pyplot.show()

#3.3
tv = TfidfVectorizer(vocabulary=vocab)
Y = tv.fit_transform(all_text)
newArray = Y.toarray()


tvmatrix = pd.DataFrame(newArray.transpose(), index = vocab, columns=play_names)

#3.4
tvpca = PCA(n_components=2)
tvresult = tvpca.fit_transform(tvmatrix.transpose())
pyplot.scatter(tvresult[:, 0], tvresult[:, 1])
for i, word in enumerate(tvmatrix):
	pyplot.annotate(word, xy=(tvresult[i, 0], tvresult[i, 1]))
pyplot.show()

#3.5
comatrix_model = CountVectorizer(vocabulary=vocab) 
X = comatrix_model.fit_transform(all_text).toarray()
dotprod = X.transpose().dot(X)

comatrix = pd.DataFrame(dotprod, index = vocab, columns = vocab)



#3.6
comedies = [ "The Tempest", "Two Gentlemen of Verona", "Merry Wives of Windsor", "Measure for measure", "A Comedy of Errors", "Much Ado about nothing", "Loves Labours Lost", "A Midsummer nights dream", "Merchant of Venice", "As you like it", "Taming of the Shrew", "Alls well that ends well", "Twelfth Night", "A Winters Tale", "Pericles"]
histories = ["King John", "Richard III", "Richard II", "Henry IV", "Henry V", "Henry VIII", 'Henry VI Part 3', 'Henry VI Part 2', 'Henry VI Part 1']
tragedies = ["Troilus and Cressida","Coriolanus","Titus Andronicus","Romeo and Juliet","Timon of Athens","Julius Caesar","macbeth","Hamlet","King Lear","Othello","Antony and Cleopatra","Cymbeline"]


def computeVector (words):
    text = "".join(words)
    vectors = [comatrix[word].values for word in nltk.word_tokenize(text) if word in vocab]
    return sum(vectors)/len(vectors)
            

comedyDict = {}
for play in seperatedplays.axes[0]:
    if play in comedies:
        ret = computeVector(seperatedplays[play])
        comedyDict[play] = ret

finalComedy = [comedyDict[play] for play in comedyDict]
comedySimilarities = cosine_similarity(finalComedy)
print(sum(sum(comedySimilarities))/(15*15))


historyDict = {}
for play in seperatedplays.axes[0]:
    if play in histories:
        ret = computeVector(seperatedplays[play])
        historyDict[play] = ret

finalHistory = [historyDict[play] for play in historyDict]
historySimilarities = cosine_similarity(finalHistory)
print(sum(sum(historySimilarities))/(9*9))
 

tragedyDict = {}
for play in seperatedplays.axes[0]:
    if play in tragedies:
        ret = computeVector(seperatedplays[play])
        tragedyDict[play] = ret

finalTragedy = [tragedyDict[play] for play in tragedyDict]
tragedySimilarities = cosine_similarity(finalTragedy)
print(sum(sum(tragedySimilarities))/(12*12))



#3.7
model = Word2Vec(sentences="".join(all_text).lower(), size=100, min_count=1, workers=1)
historyVectors=[]
tragedyVectors=[]
comedyVectors=[]
for play in seperatedplays.axes[0]:
    if play in comedies:
        temp = [sum([model.wv[word] if word in model else np.array([0 for num in range(100)]) for word in sentence])/len(sentence) for sentence in seperatedplays[play]]
        comedyVectors.append(sum(temp)/len(temp))
comedySimilarities = cosine_similarity(comedyVectors)
print(sum(sum(comedySimilarities))/(15*15))

for play in seperatedplays.axes[0]:
    if play in histories:
        temp = [sum([model.wv[word] if word in model else np.array([0 for num in range(100)]) for word in sentence])/len(sentence) for sentence in seperatedplays[play]]
        historyVectors.append(sum(temp)/len(temp))
historySimilarities = cosine_similarity(historyVectors)
print(sum(sum(historySimilarities))/(9*9))

for play in seperatedplays.axes[0]:
    if play in tragedies:
        temp = [sum([model.wv[word] if word in model else np.array([0 for num in range(100)]) for word in sentence])/len(sentence) for sentence in seperatedplays[play]]
        tragedyVectors.append(sum(temp)/len(temp))
tragedySimilarities = cosine_similarity(tragedyVectors)
print(sum(sum(tragedySimilarities))/(12*12))


#3.8
charplays = review_data.groupby(4)[5].apply(lambda x: list(np.unique(x)))
totalarray=[]
chardict = {}
for actor in charplays.axes[0]:
    words = nltk.word_tokenize("".join(charplays[actor]))
    chardict[actor]  = sum([model.wv[word] if word in model else np.array([0 for num in range(100)]) for word in words])/len(words)
    totalarray.append(chardict[actor])
    
totalarray=np.array(totalarray)
charpd = pd.DataFrame(totalarray.transpose(),columns=charplays.axes[0])
    
charpca = PCA(n_components=2)
charesult = charpca.fit_transform(charpd.transpose())
pyplot.scatter(charesult[:, 0], charesult[:, 1])
for i, word in enumerate(charpd):
	pyplot.annotate(word, xy=(charesult[i, 0], charesult[i, 1]))
pyplot.show()


    
            

