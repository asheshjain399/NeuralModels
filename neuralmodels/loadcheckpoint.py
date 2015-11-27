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
from neuralmodels.models import *

'''
def loadLayers(model,layers_to_load):
	for layer_name in layers_to_load:
		model['config'][layer_name] = [eval(layer['layer'])(**layer['config']) for layer in model['config'][layer_name]]
	return model
'''

'''
def CreateSaveableModel(model,layers_to_save):
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
'''

def loadLayers(model,layers_to_load):
	for layer_name in layers_to_load:
		layers_init = []
		for layer in model['config'][layer_name]:

			if 'nested_layers' in layer['config'].keys():
				if layer['config']['nested_layers']:
					layer = loadLayers(layer,['layers'])

			layers_init.append(eval(layer['layer'])(**layer['config']))
		model['config'][layer_name] = layers_init 
	return model

def CreateSaveableModel(model,layers_to_save):
	for layerName in layers_to_save:
		layer_configs = []
		for layer in getattr(model,layerName):
			if hasattr(layer,'nested_layers'):
				if layer.nested_layers:
					layer = CreateSaveableModel(layer,['layers'])
			layer_config = layer.settings
			layer_name = layer.__class__.__name__
			weights = [p.get_value() for p in layer.params]
			layer_config['weights'] = weights
			layer_configs.append({'layer':layer_name, 'config':layer_config})
		model.settings[layerName] = layer_configs
	return model


def load(path):
	model = cPickle.load(open(path))
	model_class = eval(model['model'])  #getattr(models, model['model'])
	layer_args = ['layers']
	model = loadLayers(model,layer_args)
	model = model_class(**model['config'])
	return model


def loadDRAskeleton(path):
	model = cPickle.load(open(path))
	model_class = eval(model['model'])  #getattr(models, model['model'])

	edgeRNNs = {}
	for k in model['config']['edgeRNNs'].keys():
		layerlist = model['config']['edgeRNNs'][k]
		edgeRNNs[k] = []
		for layer in layerlist:
			if 'nested_layers' in layer['config'].keys():
				if layer['config']['nested_layers']:
					layer = loadLayers(layer,['layers'])
			edgeRNNs[k].append(eval(layer['layer'])(**layer['config']))
		#edgeRNNs[k] = [eval(layer['layer'])(**layer['config']) for layer in layerlist]
	model['config']['edgeRNNs'] = edgeRNNs

	nodeRNNs = {}
	for k in model['config']['nodeRNNs'].keys():
		layerlist = model['config']['nodeRNNs'][k]
		nodeRNNs[k] = []
		for layer in layerlist:
			if 'nested_layers' in layer['config'].keys():
				if layer['config']['nested_layers']:
					layer = loadLayers(layer,['layers'])
			nodeRNNs[k].append(eval(layer['layer'])(**layer['config']))
		#nodeRNNs[k] = [eval(layer['layer'])(**layer['config']) for layer in layerlist]
	model['config']['nodeRNNs'] = nodeRNNs
	return model,model_class

def loadmultipleDRA(path1,path2,swap_edgernn,swap_edgernn2,swap_nodernn,swap_nodernn2):
	[model,model_class] = loadDRAskeleton(path1)
	[model_2,model_class_2] = loadDRAskeleton(path2)

	key1 = model['config']['edgeRNNs'].keys()
	key2 = model_2['config']['edgeRNNs'].keys()

	for en,en2 in zip(swap_edgernn,swap_edgernn2):
	
		e1 = en
		if not en in key1:
			kk = en.split('_')
			e1 = kk[1] + '_' + kk[0]
	
		e2 = en2
		if not en2 in key2:
			kk = en2.split('_')
			e2 = kk[1] + '_' + kk[0]
		
		model['config']['edgeRNNs'][e1] = model_2['config']['edgeRNNs'][e2]
	
	for en,en2 in zip(swap_nodernn,swap_nodernn2):
		model['config']['nodeRNNs'][en] = model_2['config']['nodeRNNs'][en2]

	model = model_class(**model['config'])
	return model

def loadDRA(path):
	model = cPickle.load(open(path))
	model_class = eval(model['model'])  #getattr(models, model['model'])

	edgeRNNs = {}
	for k in model['config']['edgeRNNs'].keys():
		layerlist = model['config']['edgeRNNs'][k]
		edgeRNNs[k] = []
		for layer in layerlist:
			if 'nested_layers' in layer['config'].keys():
				if layer['config']['nested_layers']:
					layer = loadLayers(layer,['layers'])
			edgeRNNs[k].append(eval(layer['layer'])(**layer['config']))
		#edgeRNNs[k] = [eval(layer['layer'])(**layer['config']) for layer in layerlist]
	model['config']['edgeRNNs'] = edgeRNNs

	nodeRNNs = {}
	for k in model['config']['nodeRNNs'].keys():
		layerlist = model['config']['nodeRNNs'][k]
		nodeRNNs[k] = []
		for layer in layerlist:
			if 'nested_layers' in layer['config'].keys():
				if layer['config']['nested_layers']:
					layer = loadLayers(layer,['layers'])
			nodeRNNs[k].append(eval(layer['layer'])(**layer['config']))
		#nodeRNNs[k] = [eval(layer['layer'])(**layer['config']) for layer in layerlist]
	model['config']['nodeRNNs'] = nodeRNNs
	model = model_class(**model['config'])
	return model
	
def loadSharedRNNVectors(path):
	model = cPickle.load(open(path))
	model_class = eval(model['model']) #getattr(models, model['model'])
	layer_args = ['shared_layers','layer_1','layer_2','layer_1_output','layer_2_output']
	model = loadLayers(model,layer_args)
	model = model_class(**model['config'])
	return model

def loadSharedRNN(path):
	model = cPickle.load(open(path))
	model_class = eval(model['model']) #getattr(models, model['model'])
	layer_args = ['shared_layers','layer_1','layer_2']
	model = loadLayers(model,layer_args)
	model = model_class(**model['config'])
	return model

def loadSharedRNNOutput(path):
	model = cPickle.load(open(path))
	model_class = eval(model['model']) #getattr(models, model['model'])
	layer_args = ['shared_layers','layer_1','layer_2','layer_1_detection','layer_1_anticipation','layer_2_detection','layer_2_anticipation']
	model = loadLayers(model,layer_args)
	model = model_class(**model['config'])
	return model

def loadMultipleRNNsCombined(path):
	model = cPickle.load(open(path))
	model_class = eval(model['model']) #getattr(models, model['model'])

	temp_layer = []
	for rnn_layer in model['config']['rnn_layers']:
		temp_layer.append([eval(layer['layer'])(**layer['config']) for layer in rnn_layer])
	model['config']['rnn_layers'] = temp_layer
	layer_args = ['combined_layer']
	model = loadLayers(model,layer_args)
	model = model_class(**model['config'])
	return model

def save(model, path):
	sys.setrecursionlimit(10000)
	layer_args = ['layers']
	model = CreateSaveableModel(model,layer_args)
	serializable_model = {'model':model.__class__.__name__, 'config':model.settings}
	cPickle.dump(serializable_model, open(path, 'wb'))

def saveSharedRNNVectors(model, path):
	sys.setrecursionlimit(10000)
	layer_args = ['shared_layers','layer_1','layer_2','layer_1_output','layer_2_output']
	model = CreateSaveableModel(model,layer_args)
	serializable_model = {'model':model.__class__.__name__, 'config':model.settings}
	cPickle.dump(serializable_model, open(path, 'wb'))

def saveSharedRNN(model, path):
	sys.setrecursionlimit(10000)
	layer_args = ['shared_layers','layer_1','layer_2']
	model = CreateSaveableModel(model,layer_args)
	serializable_model = {'model':model.__class__.__name__, 'config':model.settings}
	cPickle.dump(serializable_model, open(path, 'wb'))

def saveDRA(model,path):
	sys.setrecursionlimit(10000)

	edgeRNNs = getattr(model,'edgeRNNs')
	edgeRNN_saver = {}
	for k in edgeRNNs.keys():
		layer_configs = []
		for layer in edgeRNNs[k]:
			if hasattr(layer,'nested_layers'):
				if layer.nested_layers:
					layer = CreateSaveableModel(layer,['layers'])
			layer_config = layer.settings
			layer_name = layer.__class__.__name__
			weights = [p.get_value() for p in layer.params]
			layer_config['weights'] = weights
			layer_configs.append({'layer':layer_name, 'config':layer_config})
		edgeRNN_saver[k] = layer_configs
	model.settings['edgeRNNs'] = edgeRNN_saver

	nodeRNNs = getattr(model,'nodeRNNs')
	nodeRNN_saver = {}
	for k in nodeRNNs.keys():
		layer_configs = []
		for layer in nodeRNNs[k]:
			if hasattr(layer,'nested_layers'):
				if layer.nested_layers:
					layer = CreateSaveableModel(layer,['layers'])
			layer_config = layer.settings
			layer_name = layer.__class__.__name__
			weights = [p.get_value() for p in layer.params]
			layer_config['weights'] = weights
			layer_configs.append({'layer':layer_name, 'config':layer_config})
		nodeRNN_saver[k] = layer_configs
	model.settings['nodeRNNs'] = nodeRNN_saver
	serializable_model = {'model':model.__class__.__name__, 'config':model.settings}
	cPickle.dump(serializable_model, open(path, 'wb'))

def saveSharedRNNOutput(model, path):
	sys.setrecursionlimit(10000)
	layer_args = ['shared_layers','layer_1','layer_2','layer_1_detection','layer_1_anticipation','layer_2_detection','layer_2_anticipation']
	model = CreateSaveableModel(model,layer_args)
	serializable_model = {'model':model.__class__.__name__, 'config':model.settings}
	cPickle.dump(serializable_model, open(path, 'wb'))

def saveMultipleRNNsCombined(model, path):
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
