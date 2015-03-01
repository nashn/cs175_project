# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 12:07:36 2015

@author: nash
"""

from tokenizer import *

import random

from nltk import FreqDist
import pylab

import nltk
import math



def tokenize_data(data_set):
    res = []
    for line in data_set:
        s = tokenize_text(line)
        
        for w in s:
            res.append(w)
    return res

def get_most_frequent_words(words, k):
    return most_frequent(words, k)
    
    
def length_barchart(text, label):
    fdist1 = FreqDist([len(w) for w in text])
    length_freq = fdist1.most_common()
    
    # get length frequency statistics
    total = len(text)
    word_length_per = {}
    #print(word_length_per)
    
    for e in length_freq:
        word_length_per[e[0]] = (100 * e[1] / total)
    #print(word_length_per)
        
    # drawing diagram
    width = 1
    #x = pylab.arange(1,len(word_length_per)+1)
    x = word_length_per.keys()
    #pylab.ylim([0,100])
    pylab.xlabel('Message Length(# characters)')
    pylab.ylabel('Percentage')
    pylab.title(label)
    pylab.bar(x, word_length_per.values(), width)
    pylab.show()
        


# modify the feature function to be our own features
#   1. statistical based features
#   2. human heuristics based features
def most_frequent_spam_features(message):
    #features = OrderedDict()
    features = {}
    #spam_words = ['a','e','i','o','u', 'sex', 'promotion']
    black_list = ['call', 'free', 'stop', 'get', 'claim', 'repli', 'text', 'go', 'txt', 'know', 'want', 'come', 'lt', 'msg', 'like', 'mobil', 'send', 'time', 'pleas', 'got', 'new', 'day', 'good', 'number', 'ok', 'love', 'contact', 'phone', 'messag', 'today', 'pound', 'prize', 'need', 'servic', 'accid', 'back', 'may', 'one', 'tri', 'think', 'see', 'entitl', 'record', 'hi', 'ye', 'urgent', 'sex', 'promotion']
    message_words = message.split(r'[^A-Za-z]')

    features['contain_number'] = 'yes' if r'[0-9]+' in message.lower() else 'no'
    features['black_list_words'] = len([w for w in message_words if w.lower() in black_list])    
    features['total_number_chars'] = len(message)

    '''
    # get all required features
    i = 1
    for w in spam:
        if i == 6:
            break
        if len(w) > 3:
            features['most_frequent_'+str(i)] = w
        i += 1
    '''
    
    # no need to return in input order
    return features

def heuristic_spam_features(message):
    #features = OrderedDict()
    features = {}    
    #black_list = ['sex', 'promotion']
    
    # get all required features 
    features['black_list_words'] = len([c for c in name if c.lower() in black_list])
    features['total_number_chars'] = len(message)
    
    # no need to return in input order
    return features


# modified from assignment #2
def test_naive_bayes(data, N,K,R):
    # modify here
    # get data from our own data set
    
    # get data set from nltk.corpus.names
    labeled_messages = ([(message, label) for message, label in data.items()])
    random.seed(R)    
    random.shuffle(labeled_messages)
    
    
    # modify the feature function to be our own features
    #   1. statistical based features
    #   2. human heuristics based features
    # using predefined function to get all features
    featuresets = [(most_frequent_spam_features(m), l) for (m, l) in labeled_messages]
    
    
    #---------------------------------------------------
    #   no need to touch the below parts
    #---------------------------------------------------
    
    # Using the first N elements of featuresets for training, 
    # and the remainder for testing, train a Naive Bayes classifier, 
    # using nltk.NaiveBayesClassifier.train()
    train_set, test_set = featuresets[:N], featuresets[N:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    # Compute the accuracy of your classifier on the test data and 
    # return the accuracy as a percentage, 
    # i.e., a raw float taking values between 0 and 100, e.g., 75.34..... 
    accuracy = 100 * nltk.classify.accuracy(classifier, test_set)
    
    # Also have the function print out the K most informative features to the screen using the function 
    classifier.show_most_informative_features(K)
    
    return accuracy
    

# modified from assignment #2
def cross_validate_naive_Bayes(data, v, r):
    # modify here
    # get data from our own data set
   # get data set from nltk.corpus.names
    labeled_messages = ([(message, label) for message, label in data.items()])
    random.seed(r)    
    random.shuffle(labeled_messages)
    
    
    # modify the feature function to be our own features
    #   1. statistical based features
    #   2. human heuristics based features
    # using predefined function to get all features
    featuresets = [(most_frequent_spam_features(m), l) for (m, l) in labeled_messages]
    
    
    #---------------------------------------------------
    #   no need to touch the below parts
    #---------------------------------------------------
    
    accuracy_list = [0] * v
    total = len(featuresets)
    m = math.floor(total / v)
    
    for i in range(0, v):
        test_set = featuresets[i*m: (i+1)*m]
        train_set = featuresets[0:i*m] + featuresets[(i+1)*m:]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        accuracy_list[i] = 100 * nltk.classify.accuracy(classifier, test_set)

    return accuracy_list
    
    
    
def main():
    print('Loading processed data...')
    ham = open('./datasets/cleanup/ham_cleanup.txt', 'r', encoding='utf-8')
    spam = open('./datasets/cleanup/spam_cleanup.txt', 'r', encoding='utf-8')

    ham_raw = [w for w in ham.readlines() if not w == '\n']
    spam_raw = [w for w in spam.readlines() if not w == '\n']

    ham.close()
    spam.close()
    
    print('Finished!')
    input('Hit Enter to continue...')

    # generating barchart for the length of messages in the data set
    print('Generating barchart for the length of messages in the data set..')
    length_barchart(ham_raw, 'Raw Ham Messages')
    length_barchart(spam_raw, 'Raw Spam Messages')
    input('Hit Enter to continue...')
    

    print('Start processing raw ham messages to word tokens...')
    ham_word_list = tokenize_data(ham_raw)
    #print(ham_dict)
    print('Start processing raw spam messages to word tokens...')
    spam_word_list = tokenize_data(spam_raw)
    print('Finished!')
    input('Hit Enter to continue...')
    
    
    most_freq_ham = get_most_frequent_words(ham_word_list, 50)    
    print( '\n\n\nTop 50 frequently show up words in ham set >\n',  most_freq_ham)
    input('Hit Enter to continue...')

    most_freq_spam = get_most_frequent_words(spam_word_list, 50)    
    print( '\n\n\nTop 50 frequently show up words in spam set >\n',  most_freq_spam)
    input('Hit Enter to continue...')
    
    '''
    ham_features = most_frequent_spam_features(most_freq_ham)
    spam_features = most_frequent_spam_features(most_freq_spam)
    '''
    
    data = {}
    for m in ham_raw:
        data[m] = 'ham'
    for m in spam_raw:
        data[m] = 'spam'
    
    print(len(data))
    print('naive bayes accuracy >\n', test_naive_bayes(data, 500, 10, 1000) )
    print('cross validation rates >\n', cross_validate_naive_Bayes(data, 4, 1000) )

    '''
    print('Generating barchart for the word frequencies ..')
    top_word_frequency_graph(ham_word_list, 50, 'Ham Word List')
    top_word_frequency_graph(spam_word_list, 50, 'Spam Word List')
    input('Hit Enter to continue...')
    
    print('Start processing ham set to dictionary...')
    ham_dict = tokenize_data(ham_set)
    
    #print(ham_dict)    
    
    print('Start processing spam set to dictionary...')
    spam_dict = tokenize_data(spam_set)
    
    # form a training set for test naive bayesian
    train_dict = {}

    for k,v in ham_dict.items():
        train_dict[k] = v
    for k,v in spam_dict.items():
        train_dict[k] = v
        
    #test_naive_bayes(train_dict)
    '''
    
if __name__ == '__main__':
    main()