#!/usr/#!/usr/bin/env python
# coding: utf-8
"""

- Original Version

	Author: Susheel Suresh
	Last Modified: 04/03/2019

"""
import math
from classifier import BinaryClassifier
from utils import get_feature_vectors
import random
import numpy.matlib
import numpy as np


#-----------------Hinge-------------------
#-----------------Hinge-------------------
class BGDHinge(BinaryClassifier):
	
	def __init__(self, args):
		self.num_iter = 100
		self.bin_feats = args.bin_feats 
		self.vocab_size = args.vocab_size 
		self.lr_bgd = args.lr_bgd 
		self.w = (np.random.rand(self.vocab_size) *2) -1
		self.features = []
		self.bias = random.random()

	def randomize(self, train_data):
		tr_size = len(train_data[0])
		indices = range(tr_size)
		random.seed(5)
		random.shuffle(indices)
		train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
		return train_data
		

	def updateWeights(self, labels):
		g = np.zeros(len(self.w))
		for x in range(len(self.features)):
			label = labels[x]
			if(np.dot(self.w, self.features[x]) * label >= 1):
				pass
			else:
				featureNump = np.array(self.features[x])
				y = np.array(featureNump * label)
				g = np.subtract(g, y)
		steps = np.array(self.lr_bgd * g)
		self.w = np.add(self.w, -steps)
		return steps

		
	def fit(self, train_data):
		data = self.randomize(train_data)
		self.features =  get_feature_vectors(data[0], self.bin_feats)
		labels = data[1]
		for x in range(self.num_iter):
			steps = self.updateWeights(labels)
			self.bias = self.bias - (labels[x] * self.lr_bgd)
			if(np.any((steps <= -.001) | (steps >= .001))):
				pass
			else:
				break

		
	def predict(self, test_x):
		features = get_feature_vectors(test_x, self.bin_feats)
		output = []
		for x in range(len(features)):
			featureNump = np.array(features[x])
			if(self.bias + np.dot(self.w, featureNump) < 0):
				output.append(-1)
			else:
				output.append(1)
		return output





#-----------------HingeReg-------------------
#-----------------HingeReg-------------------



class BGDHingeReg(BinaryClassifier):
	
	def __init__(self, args):
		self.num_iter = 100
		self.bin_feats = args.bin_feats 
		self.vocab_size = args.vocab_size 
		self.lr_bgd = args.lr_bgd 
		self.w = (np.random.rand(self.vocab_size) *2) -1
		self.features = []
		self.bias = random.random()
		#reg
		self.lamb = args.lamb /self.num_iter

	def randomize(self, train_data):
		tr_size = len(train_data[0])
		indices = range(tr_size)
		random.seed(5)
		random.shuffle(indices)
		train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
		return train_data
		

	def updateWeights(self, labels):
		g = np.zeros(len(self.w))
		for x in range(len(self.features)):
			label = labels[x]
			if(np.dot(self.w, self.features[x]) * label >= 1):
				pass
			else:
				featureNump = np.array(self.features[x])
				y = np.array(featureNump * label)
				g = np.subtract(g, y)
		steps = np.array(self.lr_bgd * g)
		coef = np.array(self.w * self.lamb)
		self.w = np.add(self.w, coef)
		self.w = np.add(self.w, -steps)
		return steps

		
	def fit(self, train_data):
		data = self.randomize(train_data)
		self.features =  get_feature_vectors(data[0], self.bin_feats)
		labels = data[1]
		for x in range(self.num_iter):
			steps = self.updateWeights(labels)
			self.bias = self.bias - (labels[x] * self.lr_bgd)
			if(np.any((steps <= -.001) | (steps >= .001))):
				pass
			else:
				break

		
	def predict(self, test_x):
		features = get_feature_vectors(test_x, self.bin_feats)
		output = []
		for x in range(len(features)):
			featureNump = np.array(features[x])
			if(self.bias + np.dot(self.w, featureNump) < 0):
				output.append(-1)
			else:
				output.append(1)
		return output



#-----------------Log-------------------
#-----------------Log-------------------





class BGDLog(BinaryClassifier):
	
	def __init__(self, args):
		self.num_iter = 26
		self.bin_feats = args.bin_feats 
		self.vocab_size = args.vocab_size 
		self.lr_bgd = .2
		self.w = (np.random.rand(self.vocab_size) *2) -1
		self.features = []
		self.bias = random.random()

	def randomize(self, train_data):
		tr_size = len(train_data[0])
		indices = range(tr_size)
		random.seed(5)
		random.shuffle(indices)
		train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
		return train_data
		

	def updateWeights(self, labels):
		g = np.zeros(len(self.w))
		for x in range(len(self.features)):
			label = labels[x]
			featureNump = np.array(self.features[x])
			weightedFeat = np.multiply(featureNump,self.w)
			sigmoid = np.array(1/ (1 + np.exp(-weightedFeat)))
			s = np.array(label - sigmoid)
			g = np.subtract(g,np.multiply(s,featureNump))
		steps = np.array(self.lr_bgd * g)
		steps = steps/ self.num_iter
		self.w = np.add(self.w, -steps)
		return steps

		
	def fit(self, train_data):
		data = self.randomize(train_data)
		self.features =  get_feature_vectors(data[0], self.bin_feats)
		labels = data[1]
		for x in range(len(labels)):
			if labels[x] == -1:
				labels[x] = 0
		for x in range(self.num_iter):
			steps = self.updateWeights(labels)
			self.bias = self.bias - (labels[x] * self.lr_bgd)
			if(np.any((steps <= -.001) | (steps >= .001))):
				pass
			else:
				break

		
	def predict(self, test_x):
		features = get_feature_vectors(test_x, self.bin_feats)
		output = []
		for x in range(len(features)):
			featureNump = np.array(features[x])
			predict = self.bias + np.dot(self.w, featureNump)
			if 1/(1+math.exp(-predict)) <.05:
				output.append(-1)
			else: 
				output.append(1)
		return output












#-----------------LogReg-------------------
#-----------------LogReg-------------------





class BGDLogReg(BinaryClassifier):
    
    def __init__(self, args):
        self.num_iter = 26
        self.bin_feats = args.bin_feats 
        self.vocab_size = args.vocab_size 
        self.lr_bgd = .2
        self.w = (np.random.rand(self.vocab_size) *2) -1
        self.features = []
        self.bias = random.random()
        self.lamb = args.lamb /self.num_iter

    def randomize(self, train_data):
        tr_size = len(train_data[0])
        indices = range(tr_size)
        random.seed(5)
        random.shuffle(indices)
        train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
        return train_data
        

    def updateWeights(self, labels):
        g = np.zeros(len(self.w))
        for x in range(len(self.features)):
            label = labels[x]
            featureNump = np.array(self.features[x])
            weightedFeat = np.multiply(featureNump,self.w)
            sigmoid = np.array(1/ (1 + np.exp(-weightedFeat)))
            s = np.array(label - sigmoid)
            g = np.subtract(g,np.multiply(s,featureNump))
        steps = np.array(self.lr_bgd * g)
        coef = np.array(self.w * self.lamb)
        steps = steps/ self.num_iter
        self.w = np.add(self.w,coef)
        self.w = np.add(self.w, -steps)
        return steps

        
    def fit(self, train_data):
        data = self.randomize(train_data)
        self.features =  get_feature_vectors(data[0], self.bin_feats)
        labels = data[1]
        for x in range(len(labels)):
            if labels[x] == -1:
                labels[x] = 0
        for x in range(self.num_iter):
            steps = self.updateWeights(labels)
            self.bias = self.bias - (labels[x] * self.lr_bgd)
            if(np.any((steps <= -.001) | (steps >= .001))):
                pass
            else:
                break

        
    def predict(self, test_x):
        features = get_feature_vectors(test_x, self.bin_feats)
        output = []
        for x in range(len(features)):
            featureNump = np.array(features[x])
            predict = self.bias + np.dot(self.w, featureNump)
            if 1/(1+math.exp(-predict)) <.05:
                output.append(-1)
            else: 
                output.append(1)
        return output


