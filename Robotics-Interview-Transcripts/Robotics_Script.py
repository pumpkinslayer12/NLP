import collections
import nltk
import string
import csv
import pprint
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_corpus():
    transcript_path='H:/python_scripts/data_files/RoboticsTranscriptsCSV.txt'
    
    intervieweeTranscriptDict=dict()
    
    with open(transcript_path,'r') as file:
        csv_reader= csv.reader(file,delimiter=',',quotechar='"')
        for row in csv_reader:
            if row[1]=='Interviewee':
                if row[0] in intervieweeTranscriptDict:
                    intervieweeTranscriptDict[row[0]].extend([nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(row[2])])
                else:
                    intervieweeTranscriptDict[row[0]]=[nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(row[2])]
    return intervieweeTranscriptDict

def get_corpus_pos(dic):
    return {interviewee: nltk.pos_tag_sents(doc) for interviewee,doc in dic.items()}
    
def get_corpus_frequency(dic):
    freq= collections.defaultdict()
    for doc in list(dic.keys()):
        for sent in doc:
            for word_pos in sent:
                freq[word[0]]+=1
    return freq
    
#returns counter object of word frequencies
def get_corpus_top_word_frequency(dic,switch=1):
    #returns top 15
    freq= collections.defaultdict(int)
    for doc in dic.values():
        for sent in doc:
            if switch != 1:
                for word in sent:
                    freq[word]+=1
            else:
                for word, pos in sent:
                    freq[word]+=1
    c=collections.Counter()
    c.update(freq)
    return list(c.most_common(15))

def get_corpus_top_noun_frequency(dic):
    #returns top 10
    freq= collections.defaultdict(int)
    for doc in dic.values():
        for sent in doc:
            for word, pos in sent:
                if nltk.tag.map_tag('en-ptb','universal',pos) == 'NOUN':
                    freq[word]+=1
    c=collections.Counter()
    c.update(freq)
    return list(c.most_common(10))

def get_corpus_pronoun_frequency(dic):
    #returns top 10
    freq= collections.defaultdict(int)
    for doc in dic.values():
        for sent in doc:
            pn_buffer=list()
            for word, pos in sent:
                if pos in ['NNP','NNPS']:
                    pn_buffer.append(word)
                elif pn_buffer:
                    freq[' '.join(pn_buffer)]+=1
                    pn_buffer=list()
            if pn_buffer:
                freq[' '.join(pn_buffer)]+=1
                
    c=collections.Counter()
    c.update(freq)
    return list(c.most_common(10))
    
#lematizes words in a sentence
def get_lemmatized_corpus(pos_dic):
    lemmatizer= nltk.stem.WordNetLemmatizer()
    temp_pos_dic=dict()
    for interviewee, document in pos_dic.items():
            sent_buffer=list()
            for sentence in document:
                lemma_list= list()
                #simplify POS and Lemmatize words using WordNet
                for word, pos in sentence:
                    POS_TEST=nltk.tag.map_tag('en-ptb','universal',pos)
                    if POS_TEST=='NOUN':
                        lemma_list.append((lemmatizer.lemmatize(word,pos='n'),pos))
                    elif POS_TEST=='VERB':
                        lemma_list.append((lemmatizer.lemmatize(word,pos='v'),pos))
                    elif POS_TEST=='ADJ':
                        lemma_list.append((lemmatizer.lemmatize(word,pos='a'),pos))
                    elif POS_TEST=='ADV':
                        lemma_list.append((lemmatizer.lemmatize(word,pos='r'),pos))
                    else:
                        lemma_list.append((str.lower(word),pos))
                sent_buffer.append(lemma_list)
            temp_pos_dic[interviewee]=sent_buffer
    return temp_pos_dic

def remove_stop_words_and_lower_words(dic,switch=1):

    pruned_dict=dict()
    stopwords=nltk.corpus.stopwords.words('english')
    punc=set(string.punctuation)
    
    if switch!=1:
        for interviewee, document in dic.items():
            transcriptList=list()
            for sent in document:
                test=[str.lower(word) for word in sent if str.lower(word) not in stopwords and word not in punc]
                if test:
                    transcriptList.append(test)
            pruned_dict[interviewee]=transcriptList
    else:
        for interviewee, document in dic.items():
            transcriptList=list()
            for sent in document:
                test= [(str.lower(word), pos) for word, pos in sent if str.lower(word) not in stopwords and word not in punc]
                if test:
                    transcriptList.append(test)
            pruned_dict[interviewee]=transcriptList
    return pruned_dict

def get_lexical_diversity(dic):
    words=list()
    for document in list(dic.values()):
        for sent in document:
            for word, pos in sent:
                words.append(word)
    
    return 1.0*len(set(words))/len(words)
    
def consolidate_sent(dic):
    temp_dic=dict()
    for interviewee, doc in dic.items():
        temp_doc=list()
        for sent in doc:
            for word_tuple in sent:
                temp_doc.append(word_tuple[0])
        temp_dic[interviewee]=' '.join(temp_doc)
    return temp_dic

def write_tf_idf_similarity_matrix(dic):
    
    ord_dic=collections.OrderedDict()
    for interviewee, doc in dic.items():
        ord_dic[interviewee]=doc
    
    tfidf_vectorizer=TfidfVectorizer()
    tfidf_matrix= tfidf_vectorizer.fit_transform(tuple(ord_dic.values()))
    matrix=(tfidf_matrix*tfidf_matrix.T).A
    with open('H:/data/cos_matrix.csv','w') as file1:
        
        p=0
        for row in matrix:
            file1.write(str(list(ord_dic.keys())[p]))
            p+=1
            for column in row:
                file1.write(','+str(column))
            file1.write('\n')
    
    
corpus=get_corpus()
corpus=get_corpus_pos(corpus)
corpus=remove_stop_words_and_lower_words(corpus)
corpus=get_lemmatized_corpus(corpus)
corpus=consolidate_sent(corpus)
write_tf_idf_similarity_matrix(corpus)


#print(str(get_lexical_diversity(corpus)))
#print('****************************')

#freq=get_corpus_top_word_frequency(corpus)
#pprint.pprint(freq)
#print('****************************')

#freq=get_corpus_top_noun_frequency(corpus)
#pprint.pprint(freq)
#print('****************************')

#freq=get_corpus_pronoun_frequency(corpus)
#pprint.pprint(freq)
#print('****************************')
#print(str(lexical_diversity(corpus)))
