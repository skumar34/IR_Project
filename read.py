######################################################################################
# libraries to Import.
##import sklearn
import csv
import nltk
import twokenize as tk
import time
import datetime
import re

#import CMUTweetTagger

# For windows downlaod numpy from http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
# Copy the downloaded file to C:\Python27\Scripts
# Run cmd and run following command "pip install <name of the file downloaded>.whl"
import numpy
from numpy import random as r

from nltk.corpus import stopwords

# For installing stop_words "pip install stop_words"
from stop_words import get_stop_words
from nltk.stem import PorterStemmer

# Due to some error pos_tagger was not workingg. So used the textblob average perceptron tagger
# "pip install -U textblob textblob-aptagger"
from textblob import TextBlob as tb
#from textblob_aptagger import PerceptronTagger

###################################################################################
# debug vraiable. Set this flag to 1 to enable debugging. Else set to 0. 
__DEBUG__ = 0

# For testing purpose
def TRACE(*var):
    if(__DEBUG__ == 1):
        print(var)
        print("\n")
###################################################################################


# Event Retrieval Function
def find_events(tweet_txt):
    
    pos_tag = CMUTweetTagger.runtagger_parse(tweet_txt)
    TRACE(pos_tag)
    return 0
# Find the exact word
def find_word(text,search):
    result = re.findall('\\b'+search+'\\b', text, flags=re.IGNORECASE)
    if(len(result)> 0):
        return 1
    else:
        return 0
                        
dict_relevant_tokens = {}
dict_relevant_stm_tokens = {}
dict_relevant_stm_token_pair = {}

# Extract Only Relevant Tweets and save in a file
def relevant_tweets(text):
    for tk in text:
        stem_word = ps.stem(tk)

        if stem_word in dict_relevant_stm_token_pair:
            if(find_word(dict_relevant_stm_token_pair[stem_word],tk) == 0):
                dict_relevant_stm_token_pair[stem_word] = dict_relevant_stm_token_pair[stem_word] + " " + tk
        else:
            dict_relevant_stm_token_pair[stem_word] = tk
            
        if stem_word in dict_relevant_stm_tokens:
            dict_relevant_stm_tokens[stem_word] += 1
        else:
            dict_relevant_stm_tokens[stem_word] = 1
            
        if tk in dict_relevant_tokens:
            dict_relevant_tokens[tk] +=1
        else:
            dict_relevant_tokens[tk] = 1
        #TRACE(t)
        
    return 0

# File writer module using data as dictionary
def write_file(fname, dict_data):
    with open(fname, 'wb') as f:
        writer = csv.writer(f)
        try:
            for key in dict_data:
                writer.writerow([key, dict_data[key]])
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' %(fname, writer.line_num, e))
            return e
    
    f.close()
    return 0

##ifile  = open('F:/Srinjay/Tweet Feed/Tweet Feed/classsified.csv', "rb")
# read the tweet file
ifile  = open("classified.csv","rb")
reader = csv.reader(ifile)

# flag for extract the header in csv file 
rownum = 0

# Array for storing the tweet time
tweetTime = []
# Array for storing the tweet description
tweetDesc = []

# Array for storing the relevance status of tweets.
tweetR = []
# Array for storing the tweet tokens
tweetToken = []
tweet = ""

# Stops words taken from two different sources. One from nltk and another from stop_word package
# then combine the two.
stop_words = get_stop_words('english')
#TRACE(stop_words)

stop = stopwords.words('english') + stop_words
#TRACE(stop)

# stemmer removes the morphological affixes from the word, leaving only the word stem.
ps = PorterStemmer()

for row in reader:
    if (rownum == 0):
        header = row
    else:
        colnum = 0
        # If the row read is not a blank line or non-tweet line
        if (row[0].find("+0000")!=-1):

            TRACE(1,row[0])
            # convert all the characters to lower case
            tweet = row[0].lower()
            
            tweetR.append(row[1])

            # Add the value 10 to +0000 because the tweet text starts after 10 indexes
            brk = tweet.index('+0000')+10

            
            tweetTime.append(datetime.datetime.strptime(tweet[:brk], "%a %b %d %H:%M:%S +0000 %Y"))
            TRACE(2,tweetTime)
            
            x = tweet[brk:]

            squeeze = tk.squeezeWhitespace(x)
            TRACE(3,squeeze)
            
            normal = tk.normalizeTextForTagger(squeeze.decode('utf8'))

            TRACE(4,normal)
            
            tweetDesc.append(normal)

            TRACE(5,tweetDesc)
            
            punct_num = re.compile(r'[-.?!,":;()|0-9]')
            time_pat = re.compile("(\d{1,2}(.\d{1,2})|\d{1,2})(am|pm|AM|Am|PM|Pm)")
            date_pat = re.compile("\d{1,2}\/\d{1,2}")
            week_pat = re.compile("Sun|Mon|Tue|Wed|Thurs|Fri|Sat|sunday|monday|tuesday|wednesday|thursday|friday|saturday/",re.I)

            #TRACE(6,find_events(normal))
            
            if(time_pat.search(normal)):
                normal = normal + " timepresent"
            if(date_pat.search(normal)):
                normal = normal + " datepresent"
            if(week_pat.search(normal)):
                normal = normal + " weekpresent"

            TRACE(7,normal)
            
            normal = re.sub(time_pat, '', normal)
            normal = re.sub(date_pat, '', normal)
            normal = re.sub(week_pat, '', normal)
            normal = punct_num.sub("", normal)
            tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
            b = tokenizer.tokenize(normal)
            TRACE(8,b)

       
            
            b = [i for i in b if (i not in stop)]

            if(row[1].find("Relevant")!=-1):
                '''if(row[0].find("continue")!=-1):
                    print(row[0])'''   
                relevant_tweets(b)

            token = [ps.stem(i) for i in b]

            TRACE(9,rownum)

            tweetToken.append(token)

            TRACE(10,tweetToken)
            
            #check = input('Continue or Not:' )
            
    rownum += 1
ifile.close()

#print(dict_relevant_tokens)
fname = 'rel_tokens.csv'
write_file(fname,dict_relevant_tokens)
write_file('rel_stem_token.csv',dict_relevant_stm_tokens)
write_file('rel_stem_token_pair.csv',dict_relevant_stm_token_pair)

with open('token_stem_freq.csv', 'wb') as f:
	writer = csv.writer(f)
        try:
            for key in dict_relevant_stm_tokens:
                writer.writerow([key, dict_relevant_stm_token_pair[key],dict_relevant_stm_tokens[key]])
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' %(fname, writer.line_num, e))    
    	f.close()
    
## feature engineering

documents = []
all_words = []
tweet_non = []
tweet_rel = []
for i in range(0,len(tweetR)):
    documents.append((tweetToken[i],tweetR[i]))
    all_words.extend(tweetToken[i])
    if (tweetR[i] == 'Non-Relevant'):
        tweet_non.extend(tweetToken[i])
    else:
        tweet_rel.extend(tweetToken[i])
        
all_words_freq = nltk.FreqDist(all_words)
rel_words_freq = nltk.FreqDist(tweet_rel)
non_words_freq = nltk.FreqDist(tweet_non)

## document in terms of features - function

def find_features(document,featureset):
    words = set(document)
    features = {}
    for w in featureset:
        if w in words:
            features[w] = 1
        else:
            features[w] = 0
    return features

##ranked words according to c/n ratio with add 1 smoothing

init_features = list(all_words_freq.keys())
score_words = []
for i in init_features:
    score_words.append([float(rel_words_freq[i]+1)/float(non_words_freq[i]+1),i])
score_words = sorted(score_words, reverse=True)
scores = []
scores = [i[0] for i in score_words]
scores_mean = numpy.average(scores)
features_1 = [];

#random sample gneration 2000 - train, rest - test


a = (r.uniform(0,len(tweetToken),2000))
b = [int(i) for i in a]

accu = []
threshold = range(5,20,2)
threshold = [float(i)/10 for i in threshold]
threshold = [0.7]
for t in threshold:
    features_1 = [];
    for i in range(0,len(score_words)):
        if score_words[i][0]>t:
            features_1.append(score_words[i][1])

    feature_score1=[]
    for i in range(0,len(tweetR)):
        feature_score1.append([find_features(tweetToken[i],features_1),tweetR[i]])

    trainingset = []
    for i in b:
        trainingset.append(feature_score1[i])

    testset = [x for x in feature_score1 if x not in trainingset]


    ##naive base

    naive = nltk.NaiveBayesClassifier.train(trainingset)
    accuracy = nltk.classify.accuracy(naive,testset)
    accu.append([t,accuracy])

##clf = svm.SVC(Kernel='rbf')
##tweetY = [i[1] for i in trainingset]
##tweetX = [i[0] for i in testset]
##clf.fit(trainingset[][0],training[][1])

#Before running this code, train your classifier

#classify incoming tweet
working_tweet = tweetToken[29]
relevance = naive.classify(find_features(working_tweet,features_1))
if(relevance == 'Relevant'):
    #date extraction from
    wtd = tweetDesc[29]
    datereg = re.compile("\d{1,2}\/\d{1,2}|tomorrow|tmrw|yesterday|tonight|today/",re.I)
    tomorrowreg = re.compile("tomorrow|tmrw",re.I)
    today = re.compile("today|tonight",re.I)
    datereg2 = re.compile("\d{1,2}\/\d{1,2}",re.I)
    matchdate = re.findall(datereg,wtd)
    matchdateuni = []
    positiondate = []
    for i in range(0,len(matchdate)):
        if re.match(tomorrowreg,matchdate[i]) != None:
            matchdateuni.append(tweetTime[29] + datetime.timedelta(days=1))
        elif re.match(today,matchdate[i]) != None:
            matchdateuni.append(tweetTime[29] + datetime.timedelta(days=0))
        elif re.match(datereg2,matchdate[i]) != None:
            year = tweetTime[29].year
            newdate = matchdate[i]+"/"+str(year)
            newdate = datetime.datetime.strptime(newdate, "%m/%d/%Y")
            matchdateuni.append(newdate)
        start = tweetDesc[29].find(matchdate[i])
        end = start + len(matchdate[i])-1
        positiondate.append([start,end])
