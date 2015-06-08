import theano
import numpy as np
import cPickle
from theano import tensor as T 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math

def zero0s(shape):
	return theano.shared(value=np.zeros(shape,dtype=theano.config.floatX))

def permute(samples):
	x = np.random.permutation(samples)
	for i in range(5):
		x = x[np.random.permutation(samples)]
	return x

def load(path):
# This method is taken from Passage
	import layers
        import models
	model = cPickle.load(open(path))
	model_class = getattr(models, model['model'])
	model['config']['layers'] = [getattr(layers, layer['layer'])(**layer['config']) for layer in model['config']['layers']]
	model = model_class(**model['config'])
	return model

def save(model, path):
# This method is taken from Passage
	import sys
	sys.setrecursionlimit(10000)
	layer_configs = []
	for layer in model.layers:
		layer_config = layer.settings
		layer_name = layer.__class__.__name__
		weights = [p.get_value() for p in layer.params]
		layer_config['weights'] = weights
		layer_configs.append({'layer':layer_name, 'config':layer_config})
	model.settings['layers'] = layer_configs
	serializable_model = {'model':model.__class__.__name__, 'config':model.settings}
	cPickle.dump(serializable_model, open(path, 'wb'))

def plot_loss(lossfile):
	f = open(lossfile,'r')
	lines = f.readlines()
	f.close()
	loss = [float(l.strip()) for l in lines]
	iterations = range(len(loss))
	plt.plot(iterations,loss)
	plt.show()
