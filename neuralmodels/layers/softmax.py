from headers import *

class softmax(object):
	def __init__(self,nclass,init='uniform',weights=None,rng=None):
		self.settings = locals()
		del self.settings['self']
		self.size = nclass
		self.init = getattr(inits,init)
		self.weights = weights
		self.rng = rng
	
	def connect(self,layer_below):
		self.layer_below = layer_below
		self.inputD = layer_below.size
		self.Whv = self.init((self.inputD,self.size),rng=self.rng)
		self.bhv = zero0s((1,self.size))
		self.params = [self.bhv, self.Whv]

		if self.weights is not None:
			for param, weight in zip(self.params,self.weights):
				param.set_value(np.asarray(weight, dtype=theano.config.floatX))
		
		self.L2_sqr = (self.Whv ** 2).sum()

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

