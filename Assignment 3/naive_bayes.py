'''
Created on Sep 29, 2017

@author: aqd14
'''

from __future__ import division
import numpy as np
from scipy import misc
from scipy import sparse as sps
import matplotlib.pyplot as plt

numTokens = 2500

class MultinomialNaiveBayes():
    def __init__(self):
        self.py_pos = 0.0   # estimates the probability that a particular word in a spam email will be the k-th word in the dictionary
        self.py_neg = 0.0   # estimates the probability that a particular word in a non-spam email will be the k-th word in the dictionary
        self.phi_pos = 0.0  # the probability that any particular email will be a spam email
        
    def fit(self, train_labels, train_matrix, num_tokens):
        # Training phase
        numTrainDocs = train_labels.shape[0]
        spam_email_pos = np.where(train_labels==1)      # array-like:     The indices of spam emails
        nonspam_email_pos = np.where(train_labels==0)   # array-like:     The indices of non-spam emails
        email_word_count = np.sum(train_matrix, 1)      # array-like:     The total word count for each email
        
        # Calculate phi_k|y=1 = p(xj = k|y = 1)
        self.py_pos = (train_matrix[spam_email_pos].sum(axis=0) + 1) / (np.sum(email_word_count[spam_email_pos]) + num_tokens)
        # Calculate phi_k|y=0 = p(xj = k|y = 0) 
        self.py_neg = (train_matrix[nonspam_email_pos].sum(axis=0) + 1) / (np.sum(email_word_count[nonspam_email_pos]) + num_tokens)
        
        # prior
        self.phi_pos = np.count_nonzero(train_labels)/numTrainDocs
        
    def predict(self, test_labels, test_matrix):
        num_test_docs = test_labels.shape[0]
        log_p_pos = test_matrix.dot(np.log(self.py_pos.T)) + np.log(self.phi_pos)
        log_p_neg = test_matrix.dot(np.log(self.py_neg.T)) + np.log(1 - self.phi_pos)
        
        results = log_p_pos > log_p_neg 
        # Convert from True/False to 1/0
        return np.squeeze(np.asarray(results.astype(dtype=int)))

def train_and_test(files):
    # Extract parameters
    train_labels_f = files[0]
    train_features_f = files[1]
    test_labels_f = files[2]
    test_features_f = files[3]
    
    # Load the labels for the training set
    train_labels = np.loadtxt(train_labels_f,dtype=int)
    # Get the number of training examples from the number of labels
    numTrainDocs = train_labels.shape[0]
    # This is how many words we have in our dictionary
    # Load the training set feature information
    M = np.loadtxt(train_features_f,dtype=int)
    # Create matrix of training data
    train_matrix = sps.csr_matrix((M[:,2], (M[:,0], M[:,1])), shape=(numTrainDocs, numTokens))
    
    classifier = MultinomialNaiveBayes()
    classifier.fit(train_labels, train_matrix, numTokens)
    
    test_labels = np.loadtxt(test_labels_f, dtype=int)
    # Load the test set feature information
    N = np.loadtxt(test_features_f,dtype=int)
    # Create matrix of test data
    test_matrix = sps.csr_matrix((N[:,2], (N[:,0], N[:,1])))
    
    prediction = classifier.predict(test_labels, test_matrix)
    
    num_wrong_docs = np.sum(prediction != test_labels)
    
    print('Number of wrong classification = {0}'.format(num_wrong_docs))
    print('Fraction of wrong classification = {0}\n\n'.format(num_wrong_docs/test_labels.shape[0]))
    

def main():
    files = ['pa3data/train-labels.txt', 'pa3data/train-features.txt', 'pa3data/test-labels.txt','pa3data/test-features.txt']
    print('Working with 960-document dataset...')
    train_and_test(files)
    
    files = ['pa3data/train-labels-50.txt', 'pa3data/train-features-50.txt', 'pa3data/test-labels.txt','pa3data/test-features.txt']
    print('Working with 50-document dataset...')
    train_and_test(files)
    
    files = ['pa3data/train-labels-100.txt', 'pa3data/train-features-100.txt', 'pa3data/test-labels.txt','pa3data/test-features.txt']
    print('Working with 100-document dataset...')
    train_and_test(files)
    
    files = ['pa3data/train-labels-400.txt', 'pa3data/train-features-400.txt', 'pa3data/test-labels.txt','pa3data/test-features.txt']
    print('Working with 400-document dataset...')
    train_and_test(files)
    
if __name__ == '__main__':
    main()