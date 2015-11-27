from headers import *

class LSTM(object):
	def __init__(self,activation_str='tanh',activation_gate='sigmoid',init='uniform',
		truncate_gradient=50,size=128,weights=None,seq_output=True,rng=None,
		skip_input=False,jump_up=False,grad_clip=True,g_low=-10.0,g_high=10.0):
		
		self.settings = locals()
		del self.settings['self']
		self.activation = getattr(activations,activation_str)
		self.activation_gate = getattr(activations,activation_gate)
		self.truncate_gradient = truncate_gradient
		self.init = getattr(inits,init)
		self.uniform = getattr(inits,'uniform')
		self.size = size
		self.weights = weights
		self.seq_output = seq_output
		self.rng = rng
		self.skip_input = skip_input
		self.jump_up = jump_up
		self.g_low = g_low
		self.g_high = g_high
		self.grad_clip = grad_clip


	def connect(self,layer_below,skip_layer=None):
		self.layer_below = layer_below
		self.skip_layer=skip_layer
		self.inputD = layer_below.size
		if self.skip_layer:
			self.inputD += skip_layer.size

		self.W_i = self.init((self.inputD,self.size),rng=self.rng)
		self.W_f = self.init((self.inputD,self.size),rng=self.rng)
		self.W_o = self.init((self.inputD,self.size),rng=self.rng)
		self.W_c = self.init((self.inputD,self.size),rng=self.rng)

		self.U_i = self.init((self.size,self.size),rng=self.rng)
		self.U_f = self.init((self.size,self.size),rng=self.rng)
		self.U_o = self.init((self.size,self.size),rng=self.rng)
		self.U_c = self.init((self.size,self.size),rng=self.rng)

		self.V_i = self.uniform(self.size,rng=self.rng)
		self.V_f = self.uniform(self.size,rng=self.rng)
		self.V_o = self.uniform(self.size,rng=self.rng)

		self.b_i = zero0s((self.size)) 
		self.b_f = zero0s((self.size))
		self.b_o = zero0s((self.size))
		self.b_c = zero0s((self.size))

		self.h0 = zero0s((1,self.size))
		self.c0 = zero0s((1,self.size))

		self.params = [self.W_i, self.W_f, self.W_o, self.W_c,
			self.U_i, self.U_f, self.U_o, self.U_c,
			self.b_i, self.b_f, self.b_o, self.b_c,
			self.V_i, self.V_f, self.V_o
			]

		if self.weights is not None:
			for param, weight in zip(self.params,self.weights):
				param.set_value(np.asarray(weight, dtype=theano.config.floatX))

		self.L2_sqr = (
				(self.W_i ** 2).sum() +
				(self.W_f ** 2).sum() +
				(self.W_o ** 2).sum() +
				(self.W_c ** 2).sum() +
				(self.U_i ** 2).sum() +
				(self.U_f ** 2).sum() +
				(self.U_o ** 2).sum() +
				(self.U_c ** 2).sum() 
			)

	def recurrence(self,x_t_unclip,h_tm1_unclip,c_tm1_unclip):

		x_t = theano.gradient.grad_clip(x_t_unclip,self.g_low,self.g_high)
		h_tm1 = theano.gradient.grad_clip(h_tm1_unclip,self.g_low,self.g_high)
		c_tm1 = theano.gradient.grad_clip(c_tm1_unclip,self.g_low,self.g_high)

		i_t = self.activation_gate(T.dot(x_t, self.W_i) + T.dot(h_tm1, self.U_i) + T.dot(c_tm1, T.nlinalg.diag(self.V_i)) + T.extra_ops.repeat(self.b_i,x_t.shape[0],axis=0))
		f_t = self.activation_gate(T.dot(x_t, self.W_f) + T.dot(h_tm1, self.U_f) + T.dot(c_tm1, T.nlinalg.diag(self.V_f)) + T.extra_ops.repeat(self.b_f,x_t.shape[0],axis=0))
		c_tilda_t = self.activation(T.dot(x_t, self.W_c) + T.dot(h_tm1, self.U_c) + T.extra_ops.repeat(self.b_c,x_t.shape[0],axis=0))
		c_t = f_t * c_tm1 + i_t * c_tilda_t
		o_t = self.activation_gate(T.dot(x_t, self.W_o) + T.dot(h_tm1, self.U_o) + T.dot(c_t, T.nlinalg.diag(self.V_o)) + T.extra_ops.repeat(self.b_o,x_t.shape[0],axis=0))
		h_t = o_t * self.activation(c_t)

		return h_t,c_t

	def recurrence_efficient(self,x_i,x_f,x_c,x_o,h_tm1_unclip,c_tm1_unclip):

		h_tm1 = theano.gradient.grad_clip(h_tm1_unclip,self.g_low,self.g_high)
		c_tm1 = theano.gradient.grad_clip(c_tm1_unclip,self.g_low,self.g_high)

		i_t = self.activation_gate(x_i + T.dot(h_tm1, self.U_i) + T.dot(c_tm1, T.nlinalg.diag(self.V_i)))
		f_t = self.activation_gate(x_f + T.dot(h_tm1, self.U_f) + T.dot(c_tm1, T.nlinalg.diag(self.V_f)))
		c_tilda_t = self.activation(x_c + T.dot(h_tm1, self.U_c))
		c_t = f_t * c_tm1 + i_t * c_tilda_t
		o_t = self.activation_gate(x_o + T.dot(h_tm1, self.U_o) + T.dot(c_t, T.nlinalg.diag(self.V_o)))
		h_t = o_t * self.activation(c_t)

		return h_t,c_t

	def output(self,seq_output=True,get_cell=False):
		X = []
		if self.skip_layer:
			X = T.concatenate([self.layer_below.output(),self.skip_layer.output()],axis=2)
		else:
			X = self.layer_below.output()
		
		X_i = T.dot(X, self.W_i) + self.b_i
		X_f = T.dot(X, self.W_f) + self.b_f
		X_c = T.dot(X, self.W_c) + self.b_c
		X_o = T.dot(X, self.W_o) + self.b_o
		
		h_init = T.extra_ops.repeat(self.h0,X.shape[1],axis=0)
		c_init =  T.extra_ops.repeat(self.c0,X.shape[1],axis=0)
		[out, cells], ups = theano.scan(fn=self.recurrence_efficient,
				sequences=[X_i,X_f,X_c,X_o],
				#outputs_info=[T.extra_ops.repeat(self.h0,X.shape[1],axis=0), T.extra_ops.repeat(self.c0,X.shape[1],axis=0)],
				outputs_info=[h_init,c_init],
				n_steps=X_i.shape[0],
				truncate_gradient=self.truncate_gradient
			)
		
		''' 
		[out, cells], ups = theano.scan(fn=self.recurrence,
					sequences=[X],
					outputs_info=[T.extra_ops.repeat(self.h0,X.shape[1],axis=0), T.extra_ops.repeat(self.c0,X.shape[1],axis=0)],
					n_steps=X.shape[0],
					truncate_gradient=self.truncate_gradient
				)
		'''

		if get_cell:
			return cells

		if seq_output:
			return out
			# dim = T x N x self.size 
		else:
			return out[-1]
			# dim = N x self.size 

