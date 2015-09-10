from headers import *

class simpleRNN(object):
	def __init__(self,activation_str='tanh',init='orthogonal',truncate_gradient=50,size=128,weights=None,seq_output=True,temporal_connection=True,rng=None,skip_input=False,jump_up=False):
		self.settings = locals()
		del self.settings['self']
		self.activation = getattr(activations,activation_str)
		self.truncate_gradient = truncate_gradient
		self.size = size
		self.init = getattr(inits,init)
		self.weights = weights
		self.seq_output = seq_output
		self.temporal_connection = temporal_connection
		self.rng = rng
		self.skip_input = skip_input
		self.jump_up = jump_up

	def connect(self,layer_below):
		self.layer_below = layer_below
		self.inputD = layer_below.size
		self.Wuh = self.init((self.inputD,self.size),rng=self.rng)
		self.Whh = self.init((self.size,self.size),rng=self.rng)
		self.buh = zero0s((1,self.size))
		self.h0 = zero0s((1,self.size))
		self.params = [self.Wuh, self.Whh, self.buh]
		
		if not self.temporal_connection:
			self.params = [self.Wuh, self.buh]

		if self.weights is not None:
			for param, weight in zip(self.params,self.weights):
				param.set_value(np.asarray(weight, dtype=theano.config.floatX))
		
		self.L2_sqr = (self.Wuh ** 2).sum() + (self.Whh ** 2).sum()

		if not self.temporal_connection:
			self.L2_sqr = (self.Wuh ** 2).sum()

	def recurrence(self,x_t,h_tm1):
		if self.temporal_connection:
			h_t = self.activation(T.dot(x_t, self.Wuh) + T.dot(h_tm1, self.Whh) + T.extra_ops.repeat(self.buh,x_t.shape[0],axis=0))
			return h_t
		else:
			h_t = self.activation(T.dot(x_t, self.Wuh) + T.extra_ops.repeat(self.buh,x_t.shape[0],axis=0))
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
		if self.seq_output:
			return forward_pass
			# dim = T x N x self.size 
		else:
			return forward_pass[-1]
			# dim = N x self.size 

