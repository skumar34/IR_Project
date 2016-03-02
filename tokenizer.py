import csv as csv
import os
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer

cwd = os.getcwd()

tknzr = TweetTokenizer()
rknzr = RegexpTokenizer(r'\w+')
reader = csv.reader(open('sample_list_of_tweets.csv','rU'), delimiter= ",")
for line in reader:
	tokens = rknzr.tokenize(line[0].decode("ISO-8859-1"))
	tokens = [i for i in tokens if i.isalpha()== True]
	print tokens
