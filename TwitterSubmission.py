from twython import Twython
import nltk
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import everygrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated
from nltk.lm import MLE
from nltk import flatten
import re
from nltk.tokenize import TweetTokenizer 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from nltk.corpus import stopwords
from nltk import FreqDist



APP_KEY = ### fill in here
APP_SECRET = ### fill in here
twitter = Twython(APP_KEY, APP_SECRET)

Training = []
Test = []
TrainingText = ""
TestingText = ""
TrainingText2 = ""
TestingText2 = ""

for i in range(90):
    results = twitter.search(q='covid', result_type='recent', lang ='en', count = 100, tweet_mode="extended")
    all_tweets = results['statuses']
    for tweet in all_tweets:
        tweet = tweet['full_text']
        TrainingText+= (tweet+"HOLDERHOLDER")
        TrainingText2+= tweet
        temp = tweet.lower() 
        temp = temp.replace("...", "")
        temp = temp.replace(":", "")
        temp = temp.replace("\\n", "")
        temp = temp.replace("rt", "")
        temp = re.sub("[^\w\s\d.!?;']", "", temp)
        Training+=[list(nltk.word_tokenize(sent)) for sent in nltk.sent_tokenize(temp)]
        
for i in range(10):
    results = twitter.search(q='covid', result_type='recent', lang ='en', count = 100, tweet_mode="extended")
    all_tweets = results['statuses']
    for tweet in all_tweets:
        tweet = tweet['full_text']
        TestingText+= (tweet+"HOLDERHOLDER")
        TestingText2+= tweet
        temp = tweet.lower() 
        temp = temp.replace("...", "")
        temp = temp.replace(":", "")
        temp = temp.replace("\\n", "")
        temp = temp.replace("rt", "")
        temp = re.sub("[^\w\s\d.!?;']", "", temp)
        Test+=[list(nltk.word_tokenize(sent)) for sent in nltk.sent_tokenize(temp)]
        
filename = "TestSet" + ".json"
with open(filename, 'w') as outfile:
    json.dump(TestingText outfile) 
filename = "TrainSet" + ".json"
with open(filename, 'w') as outfile:
    json.dump(TrainingText, outfile) 
        
train_data, padded_sents = padded_everygram_pipeline(1, Training)
OneGram = KneserNeyInterpolated(1) 
#OneGram = MLE(1)
OneGram.fit(train_data, padded_sents)

train_data, padded_sents = padded_everygram_pipeline(2, Training)
TwoGram = KneserNeyInterpolated(2) 
#TwoGram = MLE(2)
TwoGram.fit(train_data, padded_sents)

train_data, padded_sents = padded_everygram_pipeline(3, Training)
ThreeGram = KneserNeyInterpolated(3) 
#ThreeGram = MLE(3)
ThreeGram.fit(train_data, padded_sents)



print(OneGram.perplexity(Test))
print(TwoGram.perplexity(Test))
print(ThreeGram.perplexity(Test))

for i in range(10):
    first = "<s>"
    ret = ""
    for i in range(20):
        ret += first + " "
        first = OneGram.generate(text_seed=first)
    print(ret)
for i in range(10):
    first = "<s>"
    ret = ""
    while first != "</s>":
        ret += first + " "
        first = TwoGram.generate(text_seed=first)
    print(ret)
for i in range(10):
    first = "<s>"
    ret = ""
    while first != "</s>":
        ret += first + " "
        first = ThreeGram.generate(text_seed=first)
    print(ret)

positive = ""
negative = ""
All_tweets = TrainingText.split("HOLDERHOLDER") + TestingText.split("HOLDERHOLDER")
for tweet in All_tweets:
    if(SentimentIntensityAnalyzer().polarity_scores(tweet)['compound']) >= 0.5:
        positive += tweet
    elif(SentimentIntensityAnalyzer().polarity_scores(tweet)['compound']) <= 0.5:
        negative += tweet

stop_words = set(stopwords.words('english'))  
positiveWords = [w for w in nltk.word_tokenize(positive) if not w in stop_words] 
negativeWords = [w for w in nltk.word_tokenize(negative) if not w in stop_words]   

pDist = nltk.FreqDist(positiveWords)
nDist = nltk.FreqDist(negativeWords)

print(pDist.most_common())
print(nDist.most_common())
