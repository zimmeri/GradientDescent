#!/usr/bin/env python
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
class SGDHinge(BinaryClassifier):
	
	def __init__(self, args):
		self.num_iter = 7500
		self.bin_feats = args.bin_feats 
		self.vocab_size = args.vocab_size 
		self.lr_sgd = .1
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
		x = int(math.floor(len(self.features) * random.random() ))
		label = labels[x]
		if(np.dot(self.w, self.features[x]) * label >= 1):
			pass
		else:
			self.bias = self.bias + (label *self.lr_sgd)
			featureNump = np.array(self.features[x])
			y = np.array(featureNump * -label)
			steps = np.array(self.lr_sgd * y)
			self.w = np.add(self.w, -steps)


		
	def fit(self, train_data):
		data = self.randomize(train_data)
		self.features =  get_feature_vectors(data[0], self.bin_feats)
		labels = data[1]
		for x in range(self.num_iter):
			self.updateWeights(labels)

		
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



class SGDHingeReg(BinaryClassifier):
	
	def __init__(self, args):
		self.num_iter = 7500
		self.bin_feats = args.bin_feats 
		self.vocab_size = args.vocab_size 
		self.lr_sgd = .1
		self.w = (np.random.rand(self.vocab_size) *2) -1
		self.features = []
		self.bias = random.random()
		self.lamb = .002

	def randomize(self, train_data):
		tr_size = len(train_data[0])
		indices = range(tr_size)
		random.seed(5)
		random.shuffle(indices)
		train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
		return train_data
		

	def updateWeights(self, labels):
		g = np.zeros(len(self.w))
		x = int(math.floor(len(self.features) * random.random() ))
		label = labels[x]
		if(np.dot(self.w, self.features[x]) * label >= 1):
			pass
		else:
			self.bias = self.bias + (label *self.lr_sgd)
			coef = np.multiply(self.w, self.lamb)
			featureNump = np.array(self.features[x])
			y = np.array(featureNump * -label)
			steps = np.array(self.lr_sgd * y)
			self.w = np.add(coef, self.w)
			self.w = np.add(self.w, -steps)


		
	def fit(self, train_data):
		data = self.randomize(train_data)
		self.features =  get_feature_vectors(data[0], self.bin_feats)
		labels = data[1]
		for x in range(self.num_iter):
			self.updateWeights(labels)

		
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





class SGDLog(BinaryClassifier):
	
	def __init__(self, args):
		self.num_iter = 120
		self.bin_feats = args.bin_feats 
		self.vocab_size = args.vocab_size 
		self.lr_sgd = .1
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
		for x in range(len(self.features)):
			g = np.zeros(len(self.w))
			label = labels[x]
			featureNump = np.array(self.features[x])
			weightedFeat = np.multiply(featureNump,self.w)
			sigmoid = np.array(1/ (1 + np.exp(-weightedFeat)))
			s = np.array(label - sigmoid)
			g = np.subtract(g,np.multiply(s,featureNump))
			steps = np.array(self.lr_sgd * g)
			self.w = np.add(self.w, -steps)
			if label == 0:
				self.bias = self.bias + self.lr_sgd/ self.num_iter
			else:
				self.bias = self.bias - self.lr_sgd/ self.num_iter

		
	def fit(self, train_data):
		data = self.randomize(train_data)
		self.features =  get_feature_vectors(data[0], self.bin_feats)
		labels = data[1]
		for x in range(len(labels)):
			if labels[x] == -1:
				labels[x] = 0
		for x in range(self.num_iter):
			self.updateWeights(labels)

		
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





class SGDLogReg(BinaryClassifier):
	
	def __init__(self, args):
		self.num_iter = 150
		self.bin_feats = args.bin_feats 
		self.vocab_size = args.vocab_size 
		self.lr_sgd = .1
		self.w = (np.random.rand(self.vocab_size) *2) -1
		self.features = []
		self.bias = random.random()
		self.lamb = args.lamb

	def randomize(self, train_data):
		tr_size = len(train_data[0])
		indices = range(tr_size)
		random.seed(5)
		random.shuffle(indices)
		train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
		return train_data
		

	def updateWeights(self, labels):
		for x in range(len(self.features)):
			g = np.zeros(len(self.w))
			label = labels[x]
			featureNump = np.array(self.features[x])
			weightedFeat = np.multiply(featureNump,self.w)
			sigmoid = np.array(1/ (1 + np.exp(-weightedFeat)))
			s = np.array(label - sigmoid)
			g = np.subtract(g,np.multiply(s,featureNump))
			steps = np.array(self.lr_sgd * g)
			steps = steps/ self.num_iter
			coef = np.multiply(steps, self.lamb)
			steps = np.subtract(steps, coef)
			self.w = np.add(self.w, -steps)
			if label == 0:
				self.bias = self.bias + self.lr_sgd/ self.num_iter
			else:
				self.bias = self.bias - self.lr_sgd/ self.num_iter

		
	def fit(self, train_data):
		data = self.randomize(train_data)
		self.features =  get_feature_vectors(data[0], self.bin_feats)
		labels = data[1]
		for x in range(len(labels)):
			if labels[x] == -1:
				labels[x] = 0
		for x in range(self.num_iter):
			self.updateWeights(labels)

		
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

