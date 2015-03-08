# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:18:52 2015

@author: domuro
"""

from tokenizer import *
import nltk
import random
import math

def tokenize_data(data_set):
    res = []
    for line in data_set:
        s = tokenize_text(line)
        
        for w in s:
            res.append(w)

    return res
    
def most_frequent_spam_features(message, word):
    features = {}
    message_words = tokenize_text(message)
    features['word'] = 'yes' if word in message_words else 'no'  
    return features

def num_spam_features(message, count):
    features = {}
    message_words = tokenize_text(message)
    #features['word'] = 'yes' if word in message_words else 'no'
    for i in range(count):
        word = deterministic_words[i]
        features[word] = 'yes' if word in message_words else 'no'
    return features

def cross_validate_naive_Bayes(data, v, r, word):
    labeled_messages = ([(message, label) for message, label in data.items()])
    random.seed(r)    
    random.shuffle(labeled_messages)

    #featuresets = [(most_frequent_spam_features(m, word), l) for (m, l) in labeled_messages]
    featuresets = [(num_spam_features(m, word), l) for (m, l) in labeled_messages]
    accuracy_list = [0] * v
    total = len(featuresets)
    m = math.floor(total / v)
    
    for i in range(0, v):
        test_set = featuresets[i*m: (i+1)*m]
        train_set = featuresets[0:i*m] + featuresets[(i+1)*m:]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        accuracy_list[i] = 100 * nltk.classify.accuracy(classifier, test_set)
    return sum(accuracy_list)/len(accuracy_list)

def main():
    print('Loading processed data...')
    ham = open('./datasets/cleanup/ham_cleanup.txt', 'r', encoding='utf-8')
    spam = open('./datasets/cleanup/spam_cleanup.txt', 'r', encoding='utf-8')

    ham_raw = [w for w in ham.readlines() if not w == '\n']
    spam_raw = [w for w in spam.readlines() if not w == '\n']

    ham.close()
    spam.close()
    
    print('Finished!')

    print('Start processing raw ham messages to word tokens...')
    ham_word_list = tokenize_data(ham_raw)
    #print(ham_dict)
    print('Start processing raw spam messages to word tokens...')
    spam_word_list = tokenize_data(spam_raw)
    print('Finished!')
    
    global most_freq_ham
    global most_freq_spam    
    
    most_freq_ham = most_frequent(ham_word_list, 100)    
    print( '\n\n\nTop 50 frequently show up words in ham set >\n',  most_freq_ham)

    most_freq_spam = most_frequent(spam_word_list, 100)    
    print( '\n\n\nTop 50 frequently show up words in spam set >\n',  most_freq_spam)
    
    data = {}
    count = 0
    for m in spam_raw:
        if (data.get(m) is None):
            count += 1
        data[m] = 'spam'
    for m in ham_raw:
        if (data.get(m) is None):
            count -= 1
        data[m] = 'ham'
        #if count == 0:
        #    break;
    
    i = 0
    j = 0
    for m in data:
        if data[m] == 'spam':
            i += 1
        else:
            j += 1
    print ("spam: ", i)
    print ("ham : ", j)

    '''
    words_to_test = list(set(most_freq_ham+most_freq_spam))
    word_accuracies = {}
    for word in words_to_test:
        word_accuracies[word] = cross_validate_naive_Bayes(data, 4, 1000, word)
    
    most_deterministic_words = open('./datasets/most_deterministic_words.txt', 'w', encoding='utf-8')

    for w in sorted(word_accuracies, key=word_accuracies.get, reverse=True):
        most_deterministic_words.write(w + ' ' + str(word_accuracies[w]) + '\n')
    '''
    
    deterministic_words_file = open('./datasets/most_deterministic_words2.txt', 'r', encoding='utf-8')
    #deterministic_words = [w for w in deterministic_words_file.readlines() if not w == '\n']
    global deterministic_words
    deterministic_words = [];
    lines = deterministic_words_file.readlines()
    for line in lines:
        words = line.split(' ');
        deterministic_words.append(words[0]);

    num_word_accuracies = {}
    for i in range(100):
        num_word_accuracies[str(i)] = cross_validate_naive_Bayes(data, 4, 1000, i)
    
    num_word_accuracies_file = open('./datasets/num_word_accuracies.txt', 'w', encoding='utf-8')
    for w in sorted(num_word_accuracies, key=num_word_accuracies.get, reverse=True):
        num_word_accuracies_file.write(w + ' ' + str(num_word_accuracies[w]) + '\n')
        

if __name__ == '__main__':
    main()
