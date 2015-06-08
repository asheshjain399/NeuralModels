'''
Layout of this package is inspired by Passage https://github.com/IndicoDataSolutions/Passage
'''

import theano
import numpy as np
from theano import tensor as T
import activations
import inits
from utils import zero0s

def theano_one_hot(idx, n):
# This method is taken from 'Passage'
	z = T.zeros((idx.shape[0], n))
	one_hot = T.set_subtensor(z[T.arange(idx.shape[0]), idx], 1)
	return one_hot

class softmax(object):
	def __init__(self,nclass,init='uniform',weights=None):
		self.settings = locals()
		del self.settings['self']
		self.size = nclass
		self.init = getattr(inits,init)
		self.weights = weights
	
	def connect(self,layer_below):
		self.layer_below = layer_below
		self.inputD = layer_below.size
		self.Whv = self.init((self.inputD,self.size))
		self.bhv = zero0s((1,self.size))
		self.params = [self.bhv, self.Whv]

		if self.weights is not None:
			for param, weight in zip(self.params,self.weights):
				param.set_value(np.asarray(weight, dtype=theano.config.floatX))
		
	def output(self,Temperature=1.0):
		X = self.layer_below.output()
		is_tensor3 = X.ndim > 2
		shape = X.shape
		
		if is_tensor3:
			X = X.reshape((shape[0]*shape[1],self.inputD))
		
		out = T.nnet.softmax((1.0/Temperature)*(T.dot(X,self.Whv) + T.extra_ops.repeat(self.bhv,X.shape[0],axis=0)))

		if is_tensor3:
			out = out.reshape((shape[0],shape[1],self.size))

		return out
		# dim = T x N x self.size 


class simpleRNN(object):
	def __init__(self,activation_str='tanh',init='orthogonal',truncate_gradient=50,size=128,weights=None):
		self.settings = locals()
		del self.settings['self']
		self.activation = getattr(activations,activation_str)
		self.truncate_gradient = truncate_gradient
		self.size = size
		self.init = getattr(inits,init)
		self.weights = weights

	def connect(self,layer_below):
		self.layer_below = layer_below
		self.inputD = layer_below.size
		self.Wuh = self.init((self.inputD,self.size))
		self.Whh = self.init((self.size,self.size))
		self.buh = zero0s((1,self.size))
		self.h0 = zero0s((1,self.size))
		self.params = [self.Wuh, self.Whh, self.buh]
	
		if self.weights is not None:
			for param, weight in zip(self.params,self.weights):
				param.set_value(np.asarray(weight, dtype=theano.config.floatX))

	def recurrence(self,x_t,h_tm1):
		h_t = self.activation(T.dot(x_t, self.Wuh) + T.dot(h_tm1, self.Whh) + T.extra_ops.repeat(self.buh,x_t.shape[0],axis=0))
		#h_t = self.activation(x_t+ T.dot(h_tm1, self.Whh))
		return h_t

	def output(self):
		X = self.layer_below.output()
		#x_in = T.dot(X, self.Wuh) #+ self.buh 	
		forward_pass, ups = theano.scan(fn=self.recurrence,
					sequences=[X],
					outputs_info=[T.extra_ops.repeat(self.h0,X.shape[1],axis=0)],
					n_steps=X.shape[0],
					truncate_gradient=self.truncate_gradient
				)

		return forward_pass
		# dim = T x N x self.size 
class LSTM(object):
	def __init__(self,activation_str='tanh',activation_gate='sigmoid',init='orthogonal',truncate_gradient=50,size=128,weights=None):
		self.settings = locals()
		del self.settings['self']
		self.activation = getattr(activations,activation_str)
		self.activation_gate = getattr(activations,activation_gate)
		self.truncate_gradient = truncate_gradient
		self.init = getattr(inits,init)
		self.size = size
		self.weights = weights

	def connect(self,layer_below):
		self.layer_below = layer_below
		self.inputD = layer_below.size

		self.W_i = self.init((self.inputD,self.size))
		self.W_f = self.init((self.inputD,self.size))
		self.W_o = self.init((self.inputD,self.size))
		self.W_c = self.init((self.inputD,self.size))

		self.U_i = self.init((self.size,self.size))
		self.U_f = self.init((self.size,self.size))
		self.U_o = self.init((self.size,self.size))
		self.U_c = self.init((self.size,self.size))

		self.b_i = zero0s((1,self.size)) 
		self.b_f = zero0s((1,self.size))
		self.b_o = zero0s((1,self.size))
		self.b_c = zero0s((1,self.size))

		self.h0 = zero0s((1,self.size))
		self.c0 = zero0s((1,self.size))

		self.params = [self.W_i, self.W_f, self.W_o, self.W_c,
			self.U_i, self.U_f, self.U_o, self.U_c,
			self.b_i, self.b_f, self.b_o, self.b_c
			]

		if self.weights is not None:
			for param, weight in zip(self.params,self.weights):
				param.set_value(np.asarray(weight, dtype=theano.config.floatX))

	def recurrence(self,x_t,h_tm1,c_tm1):
		i_t = self.activation_gate(T.dot(x_t, self.W_i) + T.dot(h_tm1, self.U_i) + T.extra_ops.repeat(self.b_i,x_t.shape[0],axis=0))
		f_t = self.activation_gate(T.dot(x_t, self.W_f) + T.dot(h_tm1, self.U_f) + T.extra_ops.repeat(self.b_f,x_t.shape[0],axis=0))
		o_t = self.activation_gate(T.dot(x_t, self.W_o) + T.dot(h_tm1, self.U_o) + T.extra_ops.repeat(self.b_o,x_t.shape[0],axis=0))
		c_tilda_t = self.activation(T.dot(x_t, self.W_c) + T.dot(h_tm1, self.U_c) + T.extra_ops.repeat(self.b_c,x_t.shape[0],axis=0))

		c_t = f_t * c_tm1 + i_t * c_tilda_t
		h_t = o_t * self.activation(c_t)

		return h_t,c_t

	def output(self):
		X = self.layer_below.output()
		[out, cells], ups = theano.scan(fn=self.recurrence,
					sequences=[X],
					outputs_info=[T.extra_ops.repeat(self.h0,X.shape[1],axis=0), T.extra_ops.repeat(self.c0,X.shape[1],axis=0)],
					n_steps=X.shape[0],
					truncate_gradient=self.truncate_gradient
				)
		return out
		
class OneHot(object):
	def __init__(self,size,weights=None):
		self.settings = locals()
		del self.settings['self']
		self.size=size
		self.input=T.lmatrix()
		self.inputD=1
		self.params=[]
		self.weights=weights

	def output(self):
		return theano_one_hot(self.input.flatten(), self.size).reshape((self.input.shape[0], self.input.shape[1], self.size))

class DenseInputFeatures(object):
	'''
	Use this layer to input dense input features
	dim = Num_examples x Feature_dimension
	'''
	def __init__(self,size,weights=None):
		self.settings = locals()
		del self.settings['self']
		self.size=size
		self.input=T.matrix(dtype=theano.config.floatX)
		self.inputD=size
		self.params=[]
		self.weights=weights
	def output(self):
		return self.input

class TemporalInputFeatures(object):
	'''
	Use this layer to input dense features for RNN
	dim = Time x Num_examples x Feature_dimension
	'''
	def __init__(self,size,weights=None):
		self.settings = locals()
		del self.settings['self']
		self.size=size
		self.input=T.tensor3(dtype=theano.config.floatX)
		self.inputD=size
		self.params=[]
		self.weights=weights
	def output(self):
		return self.input
