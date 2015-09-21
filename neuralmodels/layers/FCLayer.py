from headers import *

class FCLayer(object):
	def __init__(self,activation_str='tanh',init='orthogonal',size=128,weights=None,rng=None):
		self.settings = locals()
		del self.settings['self']
		self.activation = getattr(activations,activation_str)
		self.size = size
		self.init = getattr(inits,init)
		self.weights = weights
		self.rng = rng

	def connect(self,layer_below):
		self.layer_below = layer_below
		self.inputD = layer_below.size
		self.W = self.init((self.inputD,self.size),rng=self.rng)
		self.b = zero0s((self.size))
		self.params = [self.W, self.b]
		
		if self.weights is not None:
			for param, weight in zip(self.params,self.weights):
				param.set_value(np.asarray(weight, dtype=theano.config.floatX))
		
		self.L2_sqr = (self.W ** 2).sum() 

	def output(self,seq_output=True):
		X = self.layer_below.output(seq_output=seq_output)
		return self.activation(T.dot(X, self.W) + self.b)
