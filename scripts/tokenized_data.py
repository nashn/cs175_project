# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 12:07:36 2015

@author: nash
"""

from tokenizer import *
from nltk.book import *
import pylab


def tokenize_data(data_set):
    res = {}
    for line in data_set:
        s = tokenize_text(line)
        
        for w in s:
            if w in res.keys():
                res[w] += 1
            else:
                res[w] = 1
    return res


def graph_data(text):
    fdist1 = FreqDist([w for w in text])
    common = fdist1.most_common()
    
    # calculation
    freq_list = [p[1] for p in common]
    total = len(text)
    freq_list_per = [(100 * n / total) for n in freq_list]
    freq_list_per_cumulative = [sum(freq_list_per[0:i]) for i in range(total)]

    #drawing the diagram    
    x = pylab.arange(0, total, 1)
    #x = pylab.xticks(freq_word)
    pylab.ylim([0,100])
    pylab.plot(x, freq_list_per_cumulative)
    
    pylab.xlabel('Words')
    pylab.ylabel('Cumulative Percentage')
    pylab.title('Spam Set')
    
    #pylab.grid = True
    #pylab.savefig('test.png')
    pylab.show()
    

# modify the feature function to be our own features
#   1. statistical based features
#   2. human heuristics based features
def most_frequent_spam_features(spam):
    #features = OrderedDict()
    features = {}    
    #spam_words = ['a','e','i','o','u', 'sex', 'promotion']
    
    # get all required features
    features['first_character'] = name[0]
    features['last_character'] = name[-1]
    features['first_is_vowel'] = 'yes' if name[0].lower() in vowels else 'no'
    features['last_is_vowel'] = 'yes' if name[-1].lower() in vowels else 'no'
  
    features['total_num_vowels'] = len([c for c in name if c.lower() in vowels])

    features['total_number_chars'] = len(name)
    
    # no need to return in input order
    return features

def heuristic_spam_features(spam):
    #features = OrderedDict()
    features = {}    
    #spam_words = ['a','e','i','o','u', 'sex', 'promotion']
    
    # get all required features
    features['first_character'] = name[0]
    features['last_character'] = name[-1]
    features['first_is_vowel'] = 'yes' if name[0].lower() in vowels else 'no'
    features['last_is_vowel'] = 'yes' if name[-1].lower() in vowels else 'no'
  
    features['total_num_vowels'] = len([c for c in name if c.lower() in vowels])

    features['total_number_chars'] = len(name)
    
    # no need to return in input order
    return features


# modified from assignment #2
def test_naive_bayes():
    # modify here
    # get data from our own data set
    '''
    # get data set from nltk.corpus.names
    labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
    random.seed(R)    
    random.shuffle(labeled_names)
    '''
    
    # modify the feature function to be our own features
    #   1. statistical based features
    #   2. human heuristics based features
    # using predefined function to get all features
    featuresets = [(gender_features_new(n), gender) for (n, gender) in labeled_names]
    
    
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
def cross_validate_naive_Bayes(v, r):
    # modify here
    # get data from our own data set
    '''
    # get data set from nltk.corpus.names
    labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
    random.seed(R)    
    random.shuffle(labeled_names)
    '''
    
    # modify the feature function to be our own features
    #   1. statistical based features
    #   2. human heuristics based features
    # using predefined function to get all features
    featuresets = [(gender_features_new(n), gender) for (n, gender) in labeled_names]
    
    
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
    ham = open('./datasets/cleanup/ham_cleanup.txt', 'r', encoding='utf-8')
    spam = open('./datasets/cleanup/spam_cleanup.txt', 'r', encoding='utf-8')

    ham_set = [w for w in ham.readlines() if not w == '\n']
    spam_set = [w for w in spam.readlines() if not w == '\n']

    ham.close()
    spam.close()
    
    
    print('Start processing ham set...')
    ham_dict = tokenize_data(ham_set)
    
    #print(ham_dict)    
    
    print('Start processing spam set...')
    spam_dict = tokenize_data(spam_set)
   
    #print(spam_dict)
    
    graph_data([w for w in spam_dict.keys()])


if __name__ == '__main__':
    main()