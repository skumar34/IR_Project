# Read the csv file with tweets
# stringsAsFactors control the conversion of string to factor
tweets = read.csv("Event_tweets.csv", header = TRUE, stringsAsFactors=FALSE)

# only use the tweet text and relevance status
tweets = tweets[,1:2]
tweets = transform(tweets, Relevance = as.factor(Relevance))

# seperate the relevant and non relevant tweets
rel_tweets = subset(tweets, Relevance == "Relevant")[,1]
nonrel_tweets = subset(tweets, Relevance == "Non-Relevant")[,1]
write.csv(rel_tweets, "relevant_tweets.csv", row.names = FALSE)
write.csv(nonrel_tweets, "nonrelevant_tweets.csv", row.names = FALSE)


# read a text file 
processFile("file1.txt")

processFile = function(filepath){
  textfile = file(filepath, "r")
  while(TRUE){
    line = readLines(textfile, n = 1)
    if ( length(line) == 0 ) {
      break
    }
    print(line)
  }
  close(textfile)
}
summary(tweets)

# read all the text in a single vector
raw_text = readLines("file1.txt", encoding = "ANSI")
nchar(raw_text)


# Amazon Mechanical Turk
# It allows to break down task into small components and distribute online

# Bag of Words
# create the bag of words out of total tweets
# Preprocessing Steps : 1. Clean Up irregularities(Upper case to lower case)
#                       2. remove everything that isn't a,b,c but sometimes punctuation is meaningful
#                         like @Apple : is a message to Apple, #apple : About apple
#                       3. Remove the stop words but sometimes the two words at a time is meaningful like "Take That"
#                       4. stemming : reduce the all the words to their common stem 
#                                      Approach 1: create the stem database
#                                      Approach 2: Write rule based algorithm e.g if word ends in "ed", "ing" or "ly" , remove it
#                                     This second approach is called porter stemmer designed by martin porter.
# read the data provided tweets
# stringAsFactors = FALSE override the default true option. Always use this options in case of character inputs.
# If you downloaded and installed R in a location other than the United States,you might encounter some issues when using bag of words 
# approach . So fix is to run the following code.
Sys.setlocale("LC_ALL","C")

raw_tweet_data = read.csv("tweets.csv", header = TRUE, stringsAsFactors = FALSE)

str(raw_tweet_data)
raw_tweet_data$Negative = as.factor(raw_tweet_data$Avg <= -1)
table(raw_tweet_data$Negative)

# load the tm package
require("tm")
library(SnowballC)

# corpus : It is collection of docuements. So the Corpus function in tm treat ecah element of text vector as a separate document.
# VCorpus in tm refers to 'Volatile' corpus which means that the corpus is stored in memeory and is destroyed as soon
# R containing the object is destroyed.
# PCorpus refers to the Permanent corpus which stored outside the memory say in db.
# getSources() : Find the availble sources using this functions.
# VectorSource() : It si for only character vector.


corpus = Corpus(VectorSource(raw_tweet_data$Tweet))
corpus
corpus[[1]]

# convert all the words in the corpus to the lower case
# tm_map : It is an interface to apply transformation functions (also denoted as mappings) to corpora.
# So here the tolower transformation function is applied to each document of the corpora.
# content_transformer() is used in the tm_map because otherwise the tolower won't necessarily return TextDocuments
# in tm v0.6.0 and return error while running the documenttextmatrix. Withou it, tm_map returns the characters and 
# DocumentTermMatrix isn't sure how to handle the corpus of characters.

corpus = tm_map(corpus,content_transformer(tolower))
corpus[[1]]
corpus = tm_map(corpus, removePunctuation)
corpus[[1]]

# Show the list of the stopwords
stopwords("english")
corpus = tm_map(corpus, removeWords, c("apple", stopwords("english")))
corpus[[1]]

corpus = tm_map(corpus, stemDocument)
corpus[[1]]

# Document term matrix function : It create a matrix that describes the frequency of terms that occur in a collection of documents.
# In document term matrix rows corresponds to documents in the collection and columns corresponds to the terms.
frequencies = DocumentTermMatrix(corpus)
frequencies
# look at the documents from 1000 to 1005 and words from the 505 to 515.
inspect(frequencies[1000:1005,505:515])

# find the most frequent words from the document. So if there are a lot of terms consider for computing the model
# then it means that there are more independent variables and more the complex is the system.
# So lets remove the terms which are not occuring very often.
findFreqTerms(frequencies, lowfreq = 20)

# First input is the term document matrix and second argument is the sparsity threshhold.
# What is sparsity threshold : e.g .98 means that only keep the terms appears in 2% or more of the tweets.

sparse = removeSparseTerms(frequencies,.99)
# compare this coutput with the original frequencies term document matrix. Here with .98 sparsity threshold you will find 
# that the terms reduced from 3289 to 41 i.e 1.24% of original terms
sparse

# now convert this sparse matrix to the data frame for further analysis
tweetsSparse = as.data.frame(as.matrix(sparse))
colnames(tweetsSparse) = make.names(colnames(tweetsSparse))
tweetsSparse$Negative = raw_tweet_data$Negative

# Now separate the data into training set and test set
library(caTools)
split = sample.split(tweetsSparse$Negative, SplitRatio = 0.7)
trainSparse = subset(tweetsSparse, split == TRUE)
testSparse = subset(tweetsSparse, split == FALSE)

# Predicting Sentiments
library(rpart)
library(rpart.plot)
tweetCART = rpart(Negative ~ ., data=trainSparse, method="class")

# the tree drwan can be interpreted as that if there is word freak in the tweet then predict true i.e negative
# sentiment. If hate is not in the tweet but hate is there then predict negative sentiment. if none of the above 
# is present in the tweet then predict false i.e non negative sentiment.
prp(tweetCART)

# Now evaluate the neumeical performance of our model
predictCART = predict(tweetCART, newdata = testSparse, type="class")
table(testSparse$Negative, predictCART)

# create the token out of raw text data
tokens = strsplit(raw_text, " ")
# https://rstudio-pubs-static.s3.amazonaws.com/31867_8236987cf0a8444e962ccd2aec46d9c3.html#loading-texts