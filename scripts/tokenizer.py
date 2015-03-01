# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 12:04:16 2015

@author: nash
"""

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import *
from nltk import FreqDist

import matplotlib.pyplot as plt

frequencies = {}

def load_stopwords(path):
    f = open(path, 'r')
    res = f.read().split('\n')
    return res

def tokenize_text(line):
    tokenized_line = word_tokenize(line)
    
    stop = stopwords.words('english')
    stemmer = PorterStemmer()

    # get all word tokens from the tokenized line
    words_of_tokenized_line = [w.lower() for w in tokenized_line if w.isalpha()]
    
    # remove stopwords
    #print(stopwords)
    filtered_words_of_tokenized_line = [w for w in words_of_tokenized_line if w not in stop]

    # stemming
    filtered_words_of_tokenized_line = [stemmer.stem(w) for w in filtered_words_of_tokenized_line]    
    
    return filtered_words_of_tokenized_line
    
def most_frequent(words, K):
    # countting frequencies
    for w in words:
        if w not in frequencies:
            frequencies[w] = 1
        else:
            frequencies[w] += 1
    
    # sort the frequency dictionary
    sorted_frequencies = [n for n in sorted(frequencies, key=frequencies.get, reverse=True)]
    
    # get the first K frequent strings(words)
    res = [sorted_frequencies[i] for i in range(0, K)]
    
    return res
    

def top_word_frequency_graph(text, k, label):
    fdist1 = FreqDist([w for w in text])
    word_freq = fdist1.most_common()
    
    # get frequency statistics
    total = 0
    for w in word_freq:
        total += w[1]
    
    
    word_freq_per = {}    
    for w in word_freq:
        word_freq_per[w[0]] = (100 * w[1] / total)
    #print(word_length_per)

    # drawing diagram
    #width = 1
    #x = pylab.arange(1,len(word_length_per)+1)
    #x = word_freq_per.keys()
    #pylab.ylim([0,100])
    
def main():
    pass


if __name__ == '__main__':
    main()