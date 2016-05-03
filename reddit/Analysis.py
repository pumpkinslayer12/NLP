import nltk
import csv
import pprint
import string
import re
import gc
import numpy
import matplotlib.pyplot as plt
from random import shuffle
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn import svm
from sklearn.linear_model import SGDClassifier

path_full='C:/reddit/pruned_data.txt'
path_prepped='C:/reddit/prepped.txt'
path_state_map='C:/reddit/state_class.csv'
log_file=open('C:/reddit/log_file5k.txt','w')

def get_reddit_posts(path=''):
    import csv
    raw_dict=dict()
    print('Begin loading data')
    with open(path, 'r', encoding='utf-8') as raw_file:
        csv_reader= csv.reader(raw_file,delimiter=',',quotechar='"')
        for state, comment in csv_reader:
            if state in raw_dict:
                raw_dict[state].append(comment)
            else:
                raw_dict[state]=[comment]
    print('End loading data')
    return raw_dict
            

def get_prepped_data(dic,map_path):
    prepped_dic=dict()
    feature_state_dic=dict()
    
    tknzr=nltk.tokenize.casual.TweetTokenizer()
    stopwords=nltk.corpus.stopwords.words('english')
    stopwords.extend(['URL'])
    stopwords.extend(['Deleted'])
    pos=set(string.punctuation)
    pattern= re.compile(r"(.)\1{1,}", re.DOTALL)
    
    with open(map_path, 'r') as state_map:
        csv_reader= csv.reader(state_map,delimiter=',',quotechar='"')
        
        for state, cls in csv_reader:
            feature_state_dic[state]=cls
    
    itr=1
    print('Begin prepping data')
    for state, documents in dic.items():
        
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
                    
    print('End prepping data')
    return prepped_dic
    
def write_prepped_data(dic,path=''): 
    with open(path, 'w', encoding='utf-8') as prepped_file:
        for label, documents in dic.items():
            shuffle(documents)
            for i in range(149999):
                prepped_file.write('\"'+label+'\",\"'+documents[i]+'\"\n')
    
def get_naivebayes_model(x,y,ngram=1):
    text_clf= Pipeline([('vect', CountVectorizer(ngram_range=(1,ngram),max_features=5000)),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
    text_clf=text_clf.fit(x,y)
    
    return text_clf
    
def get_svm_model(x,y,ngram=1):
    text_clf= Pipeline([('vect', CountVectorizer(ngram_range=(1,ngram),max_features=5000)),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])
    text_clf=text_clf.fit(x,y)
    
    return text_clf

def build_holdout(data, percent):
    training_x=list()
    training_y=list()
    test_x=list()
    test_y=list()
    for label, documents in data.items():
        
        temp_y=numpy.empty(len(documents))
        temp_y.fill(int(label))
        temp_y=list(temp_y)
        
        shuffle(documents)
        
        x_tr, x_te, y_tr, y_te= cross_validation.train_test_split(documents, temp_y, test_size=percent, random_state=0)
        training_x.extend(x_tr)
        training_y.extend(y_tr)
        test_x.extend(x_te)
        test_y.extend(y_te)
    return (training_x, training_y, test_x, test_y)
    

def display_holdout_results(data, hold_out, classifier, ngram):
    tuple_pack= build_holdout(data,hold_out)
    cls=None
    if classifier=='SVM':
        cls=get_svm_model(tuple_pack[0],tuple_pack[1],ngram)
    elif classifier=='NB':
        cls=get_naivebayes_model(tuple_pack[0],tuple_pack[1],ngram)
    predictions= cls.predict(tuple_pack[2])
    print(str(metrics.classification_report(tuple_pack[3],predictions)))
    log_file.write(str(metrics.classification_report(tuple_pack[3],predictions)))
    log_file.write('\n')
    print('Confusion Matrix for '+classifier+' with holdout of '+ str(hold_out)+' of ngram '+str(ngram))
    log_file.write('Confusion Matrix for '+classifier+' with holdout of '+ str(hold_out)+' of ngram '+str(ngram)+'\n')
    
    cm=metrics.confusion_matrix(tuple_pack[3],predictions)
    print(cm)
    log_file.write(str(cm))
    log_file.write('\n')
    cm=None
    tuple_pack=None
    cls=None
    predictions=None
    gc.collect()
    
def display_cross_validation(data, folds, ngram, classifier):
    text_clf=None
    if classifier=='SVM':
        text_clf= Pipeline([('vect', CountVectorizer(ngram_range=(1,ngram),max_features=25000)),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])
    elif classifier=='NB':
        text_clf= Pipeline([('vect', CountVectorizer(ngram_range=(1,ngram),max_features=25000)),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
    tuple_pack=build_holdout(data,.1)
    print('Cross Validation Scores')
    cross_val=list(cross_validation.cross_val_score(text_clf,tuple_pack[0]+tuple_pack[2],tuple_pack[1]+tuple_pack[3],cv=folds))
    
    print(cross_val)
    
    log_file.write('Cross Validation Scores')
    log_file.write(str(cross_val))
    log_file.write('\n')
    gc.collect()

data=get_reddit_posts(path_prepped)    

for i in range(2):
    print('********20% Hold out unigram************')
    print('SVM Cross')
    log_file.write('********20% Hold out unigram************\n')
    log_file.write('SVM Cross')
    log_file.write('\n')
    display_cross_validation(data, 4, 1, 'SVM')
    
    print('NB Cross')
    display_cross_validation(data, 4, 1, 'NB')
    log_file.write('NB Cross')
    log_file.write('\n')
    
    print('********SVM******')
    log_file.write('********SVM******')
    log_file.write('\n')
    display_holdout_results(data, .2, 'SVM', 1)
    
    print('********NB******')
    log_file.write('********NB******')
    log_file.write('\n')
    display_holdout_results(data, .2, 'NB', 1)
    
    print('********10% Hold out unigram************')
    print('********SVM******')
    log_file.write('********10% Hold out unigram************\n')
    log_file.write('********SVM******')
    log_file.write('\n')
    display_holdout_results(data, .1, 'SVM', 1)
    
    print('********NB******')
    log_file.write('********NB******\n')
    display_holdout_results(data, .1, 'NB', 1)
    
    print('********20% Hold out bigram************')
    print('SVM Cross')
    log_file.write('********20% Hold out bigram************\n')
    log_file.write('SVM Cross')
    log_file.write('\n')
    display_cross_validation(data, 4, 2, 'SVM')
    
    print('NB Cross')
    log_file.write('NB Cross')
    log_file.write('\n')
    display_cross_validation(data, 4, 2, 'NB')
    print('********SVM******')
    log_file.write('********SVM******')
    log_file.write('\n')
    display_holdout_results(data, .2, 'SVM', 2)
    
    print('********NB******')
    log_file.write('********NB******')
    log_file.write('\n')
    display_holdout_results(data, .2, 'NB', 2)
    
    print('********10% Hold out biigram************')
    print('********SVM******')
    log_file.write('********10% Hold out biigram************\n')
    log_file.write('********SVM******')
    log_file.write('\n')
    display_holdout_results(data, .1, 'SVM', 2)
    
    print('********NB******')
    log_file.write('********NB******')
    log_file.write('\n')
    display_holdout_results(data, .1, 'NB', 2)


    if i ==0:
        print('******Starting full dataset******')
        log_file.write('******Starting full dataset******\n')
        data=get_prepped_data(get_reddit_posts(path_full),path_state_map)
        gc.collect()
log_file.close()