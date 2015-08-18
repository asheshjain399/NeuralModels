import theano
import numpy as np
import cPickle
from theano import tensor as T 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import pdb
import sys 
from neuralmodels.layers import *

def loadLayers(model,layers_to_load):
	import models
	for layer_name in layers_to_load:
		model['config'][layer_name] = [eval(layer['layer'])(**layer['config']) for layer in model['config'][layer_name]]
	return model

def CreateSaveableModel(model,layers_to_save):
	#from neuralmodels.layers import *
	import models
	for layerName in layers_to_save:
		layer_configs = []
		for layer in getattr(model,layerName):
			layer_config = layer.settings
			layer_name = layer.__class__.__name__
			weights = [p.get_value() for p in layer.params]
			layer_config['weights'] = weights
			layer_configs.append({'layer':layer_name, 'config':layer_config})
		model.settings[layerName] = layer_configs
	
	return model

def load(path):
	# This method is taken from Passage
	#from neuralmodels.layers import *
	import models
	model = cPickle.load(open(path))
	model_class = getattr(models, model['model'])
	layer_args = ['layers']
	model = loadLayers(model,layer_args)
	model = model_class(**model['config'])
	return model

def loadSharedRNNVectors(path):
	#from neuralmodels.layers import *
	import models
	model = cPickle.load(open(path))
	model_class = getattr(models, model['model'])
	layer_args = ['shared_layers','layer_1','layer_2','layer_1_output','layer_2_output']
	model = loadLayers(model,layer_args)
	model = model_class(**model['config'])
	return model

def loadSharedRNN(path):
	#from neuralmodels.layers import *
	import models
	model = cPickle.load(open(path))
	model_class = getattr(models, model['model'])
	layer_args = ['shared_layers','layer_1','layer_2']
	model = loadLayers(model,layer_args)
	model = model_class(**model['config'])
	return model

def loadSharedRNNOutput(path):
	#from neuralmodels.layers import *
	import models
	model = cPickle.load(open(path))
	model_class = getattr(models, model['model'])
	layer_args = ['shared_layers','layer_1','layer_2','layer_1_detection','layer_1_anticipation','layer_2_detection','layer_2_anticipation']
	model = loadLayers(model,layer_args)
	model = model_class(**model['config'])
	return model

def loadMultipleRNNsCombined(path):
	#from neuralmodels.layers import *
	import models
	model = cPickle.load(open(path))
	model_class = getattr(models, model['model'])

	temp_layer = []
	for rnn_layer in model['config']['rnn_layers']:
		#temp_layer.append([getattr(layers, layer['layer'])(**layer['config']) for layer in rnn_layer])
		temp_layer.append([eval(layer['layer'])(**layer['config']) for layer in rnn_layer])
	model['config']['rnn_layers'] = temp_layer
	layer_args = ['combined_layer']
	model = loadLayers(model,layer_args)
	model = model_class(**model['config'])
	return model

def save(model, path):
	# This method is taken from Passage
	#from neuralmodels.layers import *
	import models
	sys.setrecursionlimit(10000)
	layer_args = ['layers']
	model = CreateSaveableModel(model,layer_args)
	serializable_model = {'model':model.__class__.__name__, 'config':model.settings}
	cPickle.dump(serializable_model, open(path, 'wb'))

def saveSharedRNNVectors(model, path):
	#from neuralmodels.layers import *
	import models
	sys.setrecursionlimit(10000)
	layer_args = ['shared_layers','layer_1','layer_2','layer_1_output','layer_2_output']
	model = CreateSaveableModel(model,layer_args)
	serializable_model = {'model':model.__class__.__name__, 'config':model.settings}
	cPickle.dump(serializable_model, open(path, 'wb'))

def saveSharedRNN(model, path):
	#from neuralmodels.layers import *
	import models
	sys.setrecursionlimit(10000)
	layer_args = ['shared_layers','layer_1','layer_2']
	model = CreateSaveableModel(model,layer_args)
	serializable_model = {'model':model.__class__.__name__, 'config':model.settings}
	cPickle.dump(serializable_model, open(path, 'wb'))

def saveSharedRNNOutput(model, path):
	#from neuralmodels.layers import *
	import models
	sys.setrecursionlimit(10000)
	layer_args = ['shared_layers','layer_1','layer_2','layer_1_detection','layer_1_anticipation','layer_2_detection','layer_2_anticipation']
	model = CreateSaveableModel(model,layer_args)
	serializable_model = {'model':model.__class__.__name__, 'config':model.settings}
	cPickle.dump(serializable_model, open(path, 'wb'))

def saveMultipleRNNsCombined(model, path):
	#from neuralmodels.layers import *
	import models
	sys.setrecursionlimit(10000)	
	model.settings['rnn_layers'] = []
	for layers in getattr(model,'rnn_layers'):
		layer_configs = []
		for layer in layers:
			layer_config = layer.settings
			layer_name = layer.__class__.__name__
			weights = [p.get_value() for p in layer.params]
			layer_config['weights'] = weights
			layer_configs.append({'layer':layer_name, 'config':layer_config})
		model.settings['rnn_layers'].append(layer_configs)

	layer_args = ['combined_layer']
	model = CreateSaveableModel(model,layer_args)
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
