'''
Created on Sep 29, 2017

@author: aqd14
'''

from __future__ import division
import numpy as np
from scipy import misc
from scipy import sparse as sps
import matplotlib.pyplot as plt


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
        self.py_pos = (train_matrix[spam_email_pos].sum() + 1) / (np.sum(email_word_count[spam_email_pos]) + num_tokens)
        # Calculate phi_k|y=0 = p(xj = k|y = 0) 
        self.py_neg = (train_matrix[nonspam_email_pos].sum() + 1) / (np.sum(email_word_count[nonspam_email_pos]) + num_tokens)
        
        self.phi_pos = np.count_nonzero(train_labels)/numTrainDocs
        
    def predict(self, test_labels, test_matrix):
        num_test_docs = test_labels.shape[0]
        log_p_pos = test_matrix.dot(np.log(self.py_pos)) + np.log(self.phi_pos)
        log_p_neg = test_matrix.dot(np.log(slef.py_neg)) + np.log(1 - self.phi_pos)
        return log_p_pos > log_p_neg

def main():
    # Load the labels for the training set
    train_labels = np.loadtxt('pa3data/train-labels.txt',dtype=int)
    # Get the number of training examples from the number of labels
    numTrainDocs = train_labels.shape[0]
    # This is how many words we have in our dictionary
    numTokens = 2500
    # Load the training set feature information
    M = np.loadtxt('pa3data/train-features.txt',dtype=int)
    # Create matrix of training data
    train_matrix = sps.csr_matrix((M[:,2], (M[:,0], M[:,1])), shape=(numTrainDocs, numTokens))
    
    classifier = MultinomialNaiveBayes()
    classifier.fit(train_labels, train_matrix, numTokens)
    
    test_labels = np.loadtxt('pa3data/test-labels.txt', dtype=int)
    # Load the test set feature information
    N = np.loadtxt('pa3data/test-features.txt',dtype=int)
    # Create matrix of test data
    test_matrix = sps.csr_matrix((N[:,2], (N[:,0], N[:,1])))
    
    prediction = classifier.predict(test_labels, test_matrix)
    
    num_wrong_docs = sum(prediction != test_labels)
    
    print(num_wrong_docs)
    
if __name__ == '__main__':
    main()