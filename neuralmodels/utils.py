import theano
import numpy as np
import cPickle
from theano import tensor as T 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import pdb

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

def loadSharedRNNVectors(path):
# This method is taken from Passage
	import layers
        import models
	model = cPickle.load(open(path))
	model_class = getattr(models, model['model'])
	model['config']['shared_layers'] = [getattr(layers, layer['layer'])(**layer['config']) for layer in model['config']['shared_layers']]
	model['config']['layer_1'] = [getattr(layers, layer['layer'])(**layer['config']) for layer in model['config']['layer_1']]
	model['config']['layer_2'] = [getattr(layers, layer['layer'])(**layer['config']) for layer in model['config']['layer_2']]
	model['config']['layer_1_output'] = [getattr(layers, layer['layer'])(**layer['config']) for layer in model['config']['layer_1_output']]
	model['config']['layer_2_output'] = [getattr(layers, layer['layer'])(**layer['config']) for layer in model['config']['layer_2_output']]
	model = model_class(**model['config'])
	return model

def loadSharedRNN(path):
# This method is taken from Passage
	import layers
        import models
	model = cPickle.load(open(path))
	model_class = getattr(models, model['model'])
	#pdb.set_trace()
	model['config']['shared_layers'] = [getattr(layers, layer['layer'])(**layer['config']) for layer in model['config']['shared_layers']]
	model['config']['layer_1'] = [getattr(layers, layer['layer'])(**layer['config']) for layer in model['config']['layer_1']]
	model['config']['layer_2'] = [getattr(layers, layer['layer'])(**layer['config']) for layer in model['config']['layer_2']]
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

def saveSharedRNNVectors(model, path):
# This method is taken from Passage
	import sys
	sys.setrecursionlimit(10000)

	layer_configs = []
	for layer in model.shared_layers:
		layer_config = layer.settings
		layer_name = layer.__class__.__name__
		weights = [p.get_value() for p in layer.params]
		layer_config['weights'] = weights
		layer_configs.append({'layer':layer_name, 'config':layer_config})
	model.settings['shared_layers'] = layer_configs

	layer_configs = []
	for layer in model.layer_1:
		layer_config = layer.settings
		layer_name = layer.__class__.__name__
		weights = [p.get_value() for p in layer.params]
		layer_config['weights'] = weights
		layer_configs.append({'layer':layer_name, 'config':layer_config})
	model.settings['layer_1'] = layer_configs

	layer_configs = []
	for layer in model.layer_2:
		layer_config = layer.settings
		layer_name = layer.__class__.__name__
		weights = [p.get_value() for p in layer.params]
		layer_config['weights'] = weights
		layer_configs.append({'layer':layer_name, 'config':layer_config})
	model.settings['layer_2'] = layer_configs

	layer_configs = []
	for layer in model.layer_2_output:
		layer_config = layer.settings
		layer_name = layer.__class__.__name__
		weights = [p.get_value() for p in layer.params]
		layer_config['weights'] = weights
		layer_configs.append({'layer':layer_name, 'config':layer_config})
	model.settings['layer_2_output'] = layer_configs

	layer_configs = []
	for layer in model.layer_1_output:
		layer_config = layer.settings
		layer_name = layer.__class__.__name__
		weights = [p.get_value() for p in layer.params]
		layer_config['weights'] = weights
		layer_configs.append({'layer':layer_name, 'config':layer_config})
	model.settings['layer_1_output'] = layer_configs

	serializable_model = {'model':model.__class__.__name__, 'config':model.settings}

	cPickle.dump(serializable_model, open(path, 'wb'))

def saveSharedRNN(model, path):
# This method is taken from Passage
	import sys
	sys.setrecursionlimit(10000)

	layer_configs = []
	for layer in model.shared_layers:
		layer_config = layer.settings
		layer_name = layer.__class__.__name__
		weights = [p.get_value() for p in layer.params]
		layer_config['weights'] = weights
		layer_configs.append({'layer':layer_name, 'config':layer_config})
	model.settings['shared_layers'] = layer_configs

	layer_configs = []
	for layer in model.layer_1:
		layer_config = layer.settings
		layer_name = layer.__class__.__name__
		weights = [p.get_value() for p in layer.params]
		layer_config['weights'] = weights
		layer_configs.append({'layer':layer_name, 'config':layer_config})
	model.settings['layer_1'] = layer_configs

	layer_configs = []
	for layer in model.layer_2:
		layer_config = layer.settings
		layer_name = layer.__class__.__name__
		weights = [p.get_value() for p in layer.params]
		layer_config['weights'] = weights
		layer_configs.append({'layer':layer_name, 'config':layer_config})
	model.settings['layer_2'] = layer_configs
	serializable_model = {'model':model.__class__.__name__, 'config':model.settings}

	cPickle.dump(serializable_model, open(path, 'wb'))

def loadSharedRNNOutput(path):
# This method is taken from Passage
	import layers
        import models
	model = cPickle.load(open(path))
	model_class = getattr(models, model['model'])

	layer_args = ['shared_layers','layer_1','layer_2','layer_1_detection','layer_1_anticipation','layer_2_detection','layer_2_anticipation']
	for layer_name in layer_args:
		model['config'][layer_name] = [getattr(layers, layer['layer'])(**layer['config']) for layer in model['config'][layer_name]]
	model = model_class(**model['config'])
	return model

def saveSharedRNNOutput(model, path):
	import sys
	sys.setrecursionlimit(10000)

	layer_args = ['shared_layers','layer_1','layer_2','layer_1_detection','layer_1_anticipation','layer_2_detection','layer_2_anticipation']
	
	for layerName in layer_args:
		layer_configs = []
		for layer in getattr(model,layerName):
			layer_config = layer.settings
			layer_name = layer.__class__.__name__
			weights = [p.get_value() for p in layer.params]
			layer_config['weights'] = weights
			layer_configs.append({'layer':layer_name, 'config':layer_config})
		model.settings[layerName] = layer_configs

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
