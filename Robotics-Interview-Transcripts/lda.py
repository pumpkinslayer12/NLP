import nltk
from gensim import corpora, models, similarities
import collections
import csv
import string
import pprint

data_path='H:/data_files/RoboticsTranscriptsCSV.txt'

#open and load intial file into dictionary.
#Will perform the following preprocessing:
#remove stop words
#lower cases words
#apply part of speech to remove information other than noun, verb, adjective, and adverbs
#removes empty lists
#
#Each document will disregard interviewer subject as this is more of a
#survey of what they're talking about.
#
#Each document will also disregard anything said by the interviewer. The logic 
#is that the interviewer's purpose is nothing more than to tease information out
#of the interview subject and thus would be repetitive or not data rich.

intervieweeTranscriptDict=dict()

with open(data_path) as data_file:
    csv_reader=csv.reader(data_file, delimiter=',',quotechar='"')
    for row in csv_reader:
        if row[1]=='Interviewee':
            if row[0] in intervieweeTranscriptDict:
                intervieweeTranscriptDict[row[0]].extend([nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(row[2])])
            else:
                intervieweeTranscriptDict[row[0]]=[nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(row[2])]

intervieweeTranscriptDict={interviewee: nltk.pos_tag_sents(doc) for interviewee,doc in intervieweeTranscriptDict.items()}

transcriptList=list()

for doc in intervieweeTranscriptDict.values():
    bufferDocList=list()
    stopwords=nltk.corpus.stopwords.words('english')
    for sentence in doc:
        for word, pos in sentence:
            posTest=nltk.tag.map_tag('en-ptb','universal',pos)
            if ((str.lower(word) not in stopwords and word not in set(string.punctuation)) and (posTest=='NOUN' or posTest=='VERB' or posTest=='ADJ' or posTest=='ADV')):
                bufferDocList.append(str.lower(word))
    transcriptList.append(bufferDocList)


#*************END PREPROCESSING
#The next task is to make a bag of words, normalize the word frequencies
#using a tf-idf measure and then use lsi to hopefully get a feel for the top 5
#topics covered across all of the interviews

dictionary=corpora.Dictionary(transcriptList)
corpus = [dictionary.doc2bow(text) for text in transcriptList]

#pprint.pprint(list(dictionary.items()))

tfidf= models.TfidfModel(corpus)
corpus_tfidf=tfidf[corpus]

lda= models.LdaModel(corpus_tfidf,id2word=dictionary, num_topics=5)

for i in range(0, 5):
    temp= lda.show_topic(i,10)
    strWords=""
    for term in temp:
        strWords=strWords+str(term[0])+" "
        strWords.rstrip()
    print("Top 10 terms for topic #" + str(i) + ": "+strWords)





                
                
                