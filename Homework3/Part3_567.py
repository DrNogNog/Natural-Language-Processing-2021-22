# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import similar_cos_function
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


read_plays_csv = pd.read_csv("will_play_text.csv", sep=";", header=None)
vocabularyfile = pd.read_csv(".txt", sep=";", header=None)
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

#problem 6
comedies = ["Merry Wives of Windsor", "Measure for measure","The Tempest", "Two Gentlemen of Verona", "A Comedy of Errors", "Much Ado about nothing", "Loves Labours Lost", "A Midsummer nights dream", "Merchant of Venice", "As you like it", "Taming of the Shrew", "Alls well that ends well", "Twelfth Night", "A Winters Tale", "Pericles"]
histories = ["King John", "Richard III", "Richard II", "Henry IV", "Henry V", "Henry VIII", 'Henry VI Part 3', 'Henry VI Part 2', 'Henry VI Part 1']
tragedies = ["Romeo and Juliet","Timon of Athens","Julius Caesar","macbeth","Hamlet","King Lear","Othello","Troilus and Cressida","Coriolanus","Titus Andronicus","Antony and Cleopatra","Cymbeline"]
            
comedyarray = {}
for play in plays.axes[0]:
    if play in comedies:
        text = "".join(plays[play])
        vectors = [answer[word].values for word in nltk.word_tokenize(text) if word in vocabularyfile]
        comedyarray[play] = sum(vectors)/len(vectors)
        
comedycentral = [comedyarray[play] for play in comedyarray]
cosfunct = similar_cos_function(comedycentral)
print(sum(sum(cosfunct))/(15*15))


historyarray = {}
for play in plays.axes[0]:
    if play in histories:
        text = "".join(plays[play])
        vectors = [answer[word].values for word in nltk.word_tokenize(text) if word in vocabularyfile]
        historyarray[play] = sum(vectors)/len(vectors)

finalHistory = [historyarray[play] for play in historyarray]
functh = similar_cos_function(finalHistory)
print(sum(sum(functh))/(9*9))
 

tragedyhs = {}
for play in plays.axes[0]:
    if play in tragedies:
        text = "".join(plays[play])
        vectors = [answer[word].values for word in nltk.word_tokenize(text) if word in vocabularyfile]
        tragedyhs[play] =  sum(vectors)/len(vectors)
finalTragedy = [tragedyhs[play] for play in tragedyhs]
tra_gedy = similar_cos_function(finalTragedy)
print(sum(sum(tra_gedy))/(12*12))

#Problem 7
model = Word2Vec(sentences="".join(total).lower(), size=100, min_count=1, workers=1)
H_Vect=[]
T_Vect=[]
C_Vect=[]
for play in plays.axes[0]:
    if play in comedies:
        summationvector = [sum([model.wv[word] if word in model else np.array([0 for num in range(100)]) for word in sentence])/len(sentence) for sentence in plays[play]]
        C_Vect.append(sum(summationvector)/len(summationvector))
com_edy = similar_cos_function(C_Vect)
print(sum(sum(com_edy))/(15*15))

for play in plays.axes[0]:
    if play in histories:
        summationvector = [sum([model.wv[word] if word in model else np.array([0 for num in range(100)]) for word in sentence])/len(sentence) for sentence in plays[play]]
        H_Vect.append(sum(summationvector)/len(summationvector))
his_tory = similar_cos_function(H_Vect)
print(sum(sum(his_tory))/(9*9))

for play in plays.axes[0]:
    if play in tragedies:
        summationvector = [sum([model.wv[word] if word in model else np.array([0 for num in range(100)]) for word in sentence])/len(sentence) for sentence in plays[play]]
        T_Vect.append(sum(summationvector)/len(summationvector))
tra_gedy = similar_cos_function(T_Vect)
print(sum(sum(tra_gedy))/(12*12))

    