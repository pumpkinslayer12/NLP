import nltk
import csv
import pprint
import string
import re
import gc
import numpy
import matplotlib.pyplot as plt
import collections
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

path_full='C:/reddit/pruned_data.txt'
path_state_map='C:/reddit/state_class.csv'

raw_dict=dict()
print('Begin loading data')
with open(path_full, 'r', encoding='utf-8') as raw_file:
    csv_reader= csv.reader(raw_file,delimiter=',',quotechar='"')
    for state, comment in csv_reader:
        if state in raw_dict:
            raw_dict[state].append(comment)
        else:
            raw_dict[state]=[comment]
print('End loading data')

prepped_dic=dict()
feature_state_dic=dict()

tknzr=nltk.tokenize.casual.TweetTokenizer()
stopwords=nltk.corpus.stopwords.words('english')
stopwords.extend(['URL'])
stopwords.extend(['Deleted'])
pos=set(string.punctuation)
pattern= re.compile(r"(.)\1{1,}", re.DOTALL)

with open(path_state_map, 'r') as state_map:
    csv_reader= csv.reader(state_map,delimiter=',',quotechar='"')
    
    for state, cls in csv_reader:
        feature_state_dic[state]=cls

itr=1
print('Begin prepping data')
prepped_dic=dict()
for state, documents in raw_dict.items():
    
    for post in documents:
        if itr % 100000 == 0:
            print('Prepped: ' +str(itr))
        itr+=1
        buffer_doc_list=list()
        for word in tknzr.tokenize(post):
            word=''.join(word.split(' '))
            if(word not in stopwords and word not in pos):
                buffer_doc_list.append(word)
                
        if buffer_doc_list:
            if feature_state_dic[state] in prepped_dic:
                prepped_dic[feature_state_dic[state]].append(' '.join(buffer_doc_list))
            else:
                prepped_dic[feature_state_dic[state]]=[' '.join(buffer_doc_list)]

ord_dic=collections.OrderedDict()
for label, doc in prepped_dic.items():
    ord_dic[label]=' '.join(doc)
    
tfidf_vectorizer=TfidfVectorizer(max_features=10000)
tfidf_matrix= tfidf_vectorizer.fit_transform(tuple(ord_dic.values()))
matrix=(tfidf_matrix*tfidf_matrix.T).A
with open('C:/reddit/cos_matrix.csv','w') as file1:
    
    p=0
    for row in matrix:
        file1.write(str(list(ord_dic.keys())[p]))
        p+=1
        for column in row:
            file1.write(','+str(column))
        file1.write('\n')