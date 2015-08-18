from headers import *

class BidirectionalRNN(object):
	def __init__(self,activation_str='tanh',init='orthogonal',truncate_gradient=50,size=128,weights=None,seq_output=True):
		self.settings = locals()
		del self.settings['self']
		self.activation = getattr(activations,activation_str)
		self.truncate_gradient = truncate_gradient
		self.size = size + size
		self.size_hidden = size
		self.init = getattr(inits,init)
		self.weights = weights
		self.seq_output = seq_output

	def connect(self,layer_below):
		self.layer_below = layer_below
		self.inputD = layer_below.size
		self.Wuh = self.init((self.inputD,self.size_hidden))
		self.Whh = self.init((self.size_hidden,self.size_hidden))
		self.buh = zero0s((1,self.size_hidden))
		self.h0 = zero0s((1,self.size_hidden))
		self.Wuh_b = self.init((self.inputD,self.size_hidden))
		self.Whh_b = self.init((self.size_hidden,self.size_hidden))
		self.buh_b = zero0s((1,self.size_hidden))
		self.h0_b = zero0s((1,self.size_hidden))
		self.params = [self.Wuh, self.Whh, self.buh, self.Wuh_b, self.Whh_b, self.buh_b]
		
		if self.weights is not None:
			for param, weight in zip(self.params,self.weights):
				param.set_value(np.asarray(weight, dtype=theano.config.floatX))
		
		self.L2_sqr = (self.Wuh ** 2).sum() + (self.Whh ** 2).sum()

	def recurrence(self,x_t,h_tm1):
		h_t = self.activation(T.dot(x_t, self.Wuh) + T.dot(h_tm1, self.Whh) + T.extra_ops.repeat(self.buh,x_t.shape[0],axis=0))
		return h_t

	def recurrence_backward(self,x_t,h_tm1):
		h_t = self.activation(T.dot(x_t, self.Wuh_b) + T.dot(h_tm1, self.Whh_b) + T.extra_ops.repeat(self.buh_b,x_t.shape[0],axis=0))
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
		backward_pass, ups = theano.scan(fn=self.recurrence_backward,
					sequences=[X],
					outputs_info=[T.extra_ops.repeat(self.h0_b,X.shape[1],axis=0)],
					n_steps=X.shape[0],
					truncate_gradient=self.truncate_gradient,
					go_backwards=True
				)
		
		backward_pass = backward_pass[::-1]

		if self.seq_output:
			return T.concatenate([forward_pass,backward_pass],axis=2)
			# dim = T x N x self.size 
		else:
			return T.concatenate([forward_pass[-1],backward_pass[-1]],axis=1)
			# dim = N x self.size 
