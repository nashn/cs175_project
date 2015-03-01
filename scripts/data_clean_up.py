# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 18:22:17 2015

@author: nash
"""

data_set = {}
test_set = {}

spam_set = {}
ham_set = {}

def main():

    # path = input('Enter training data file path > ')
    print('Openning training data file...\nReading...')
    
    training_set = open('./datasets/train/smsspamcollection/SMSSpamCollection', 'r', encoding='utf-8')    
    testing_set = open('./datasets/test/SMSSpamCorpus01/english_big.txt', 'r', encoding='cp1252')
    
    
    training_data = training_set.readlines()
    
    testing_data = testing_set.readlines()
    
    for data in training_data:
        temp = data.split('\t')
        
        data_set[temp[1]] = temp[0]
        
    for data in testing_data:
        temp = data.split(',')
        
        test_set[temp[0]] = temp[1]
        
    print( len(test_set) )
    
    
    for k,v in data_set.items():
        if v == 'spam':
            spam_set[k] = v
        else:
            ham_set[k] = v

    print('Reading Finished! Here are some data...')
    print('data set size    = ', len(data_set))
    print('spam sample size = ', len(spam_set))
    print('ham sample size  = ', len(ham_set))
    
    print('store parsed results ...')
    
    spam_out = open('./datasets/spam_cleanup.txt', 'w', encoding='utf-8')
    ham_out = open('./datasets/ham_cleanup.txt', 'w', encoding='utf-8')
    
    for k, v in spam_set.items():
        spam_out.write( k + '\n' )
    
    for k, v in ham_set.items():
        ham_out.write( k + '\n' )
        
    print('Done!')
    spam_out.close()
    ham_out.close()


if __name__ == '__main__':
    main()