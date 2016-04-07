import csv
import re


relevant_vocab_list = []
def create_vocab():
	with open('Selected_relevant_token_stem_freq.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile)
		try:
			for row in reader:
				vlist = row[1].split(" ")
				for word in vlist:
					relevant_vocab_list.append(word)
		except csv.Error as e:
            		sys.exit('file %s, line %d: %s' %(fname, writer.line_num, e))
            		return e
	csvfile.close()
	return relevant_vocab_list

def find_rel_tweet(fname):

	rel_token_tweet = []
	vocablist = create_vocab()
	with open(fname, 'rb') as csvfile:
		has_header = csv.Sniffer().has_header(csvfile.read(1024))
		csvfile.seek(0)
		reader = csv.reader(csvfile)
		if(has_header):
			next(reader)		# skip the header row
		
		try:	
			with open('tweet_txt_file.csv','wb') as writefile:
				writer = csv.writer(writefile) 
				for row in reader:
					tweet_txt = row[0]
					tweet_txt = tweet_txt.lower()
					#print(tweet_txt)
					#check = input('Continue or Not:' )
					tweet_txt = tweet_txt.split(" ")
					twt_rel_words = []
					for tk in tweet_txt:
						if tk in vocablist:
							twt_rel_words.append(tk)
					writer.writerow([tweet_txt,' '.join(twt_rel_words)])
					#print(' '.join(tweet_txt),' '.join(twt_rel_words))			
			writefile.close()
		except csv.Error as e:
            		sys.exit('file %s, line %d: %s' %(fname, writer.line_num, e))
            		return e
	csvfile.close()
	return 0

#print(create_vocab())
'''with open('event_vocab_2.csv','wb') as csvfile:
	wtr = csv.writer(csvfile)
	for tk in relevant_vocab_list:
		wtr.writerow([tk])
	csvfile.close()'''
find_rel_tweet('relevant_tweets.csv')
	
#print(' '.join(relevant_vocab_list))
