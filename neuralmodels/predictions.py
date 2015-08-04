import numpy as np
import random 

def OutputMaxProb(X):
	return np.argmax(X,axis=1)

def OutputActionThresh(X,default_action=1):
	with open('settings.py', 'r') as f:
		exec(f.read())

	outmax = np.argmax(X,axis=1)
	outprob = np.max(X,axis=1)
	for i in range(len(outmax)):
		if (not outmax[i] == default_action) and outprob[i] < OUTPUT_THRESH:
			outmax[i] = default_action
	return outmax

def OutputSampleFromDiscrete(X):
	labels = []
	cdf = np.zeros((X.shape[0],X.shape[1]))
	cdf[:,0] = X[:,0]
	for i in range(1,X.shape[1]):
		cdf[:,i] = cdf[:,i-1] + X[:,i]
	uniform_random = random.random()
	for i in range(cdf.shape[0]):
		for j in range(cdf.shape[1]):
			if cdf[i,j] >= uniform_random:
				labels.append(j)
				break
	return np.array(labels)
