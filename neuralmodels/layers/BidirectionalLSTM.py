from headers import *

class BidirectionalLSTM(object):
	def __init__(self,activation_str='tanh',activation_gate='sigmoid',init='orthogonal',truncate_gradient=50,size=128,weights=None,seq_output=True):
		self.settings = locals()
		del self.settings['self']
		self.activation = getattr(activations,activation_str)
		self.activation_gate = getattr(activations,activation_gate)
		self.truncate_gradient = truncate_gradient
		self.init = getattr(inits,init)
		self.size = size + size
		self.size_hidden = size
		self.weights = weights
		self.seq_output = seq_output

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

		self.W_i_b = self.init((self.inputD,self.size_hidden))
		self.W_f_b = self.init((self.inputD,self.size_hidden))
		self.W_o_b = self.init((self.inputD,self.size_hidden))
		self.W_c_b = self.init((self.inputD,self.size_hidden))

		self.U_i_b = self.init((self.size_hidden,self.size_hidden))
		self.U_f_b = self.init((self.size_hidden,self.size_hidden))
		self.U_o_b = self.init((self.size_hidden,self.size_hidden))
		self.U_c_b = self.init((self.size_hidden,self.size_hidden))

		self.b_i_b = zero0s((1,self.size_hidden)) 
		self.b_f_b = zero0s((1,self.size_hidden))
		self.b_o_b = zero0s((1,self.size_hidden))
		self.b_c_b = zero0s((1,self.size_hidden))

		self.h0_b = zero0s((1,self.size_hidden))
		self.c0_b = zero0s((1,self.size_hidden))

		self.params = [self.W_i, self.W_f, self.W_o, self.W_c,
			self.U_i, self.U_f, self.U_o, self.U_c,
			self.b_i, self.b_f, self.b_o, self.b_c,
			self.W_i_b, self.W_f_b, self.W_o_b, self.W_c_b,
			self.U_i_b, self.U_f_b, self.U_o_b, self.U_c_b,
			self.b_i_b, self.b_f_b, self.b_o_b, self.b_c_b
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

	def recurrence(self,x_t,h_tm1,c_tm1):
		i_t = self.activation_gate(T.dot(x_t, self.W_i) + T.dot(h_tm1, self.U_i) + T.extra_ops.repeat(self.b_i,x_t.shape[0],axis=0))
		f_t = self.activation_gate(T.dot(x_t, self.W_f) + T.dot(h_tm1, self.U_f) + T.extra_ops.repeat(self.b_f,x_t.shape[0],axis=0))
		o_t = self.activation_gate(T.dot(x_t, self.W_o) + T.dot(h_tm1, self.U_o) + T.extra_ops.repeat(self.b_o,x_t.shape[0],axis=0))
		c_tilda_t = self.activation(T.dot(x_t, self.W_c) + T.dot(h_tm1, self.U_c) + T.extra_ops.repeat(self.b_c,x_t.shape[0],axis=0))

		c_t = f_t * c_tm1 + i_t * c_tilda_t
		h_t = o_t * self.activation(c_t)

		return h_t,c_t

	def recurrence_backward(self,x_t,h_tm1,c_tm1):
		i_t = self.activation_gate(T.dot(x_t, self.W_i_b) + T.dot(h_tm1, self.U_i_b) + T.extra_ops.repeat(self.b_i_b,x_t.shape[0],axis=0))
		f_t = self.activation_gate(T.dot(x_t, self.W_f_b) + T.dot(h_tm1, self.U_f_b) + T.extra_ops.repeat(self.b_f_b,x_t.shape[0],axis=0))
		o_t = self.activation_gate(T.dot(x_t, self.W_o_b) + T.dot(h_tm1, self.U_o_b) + T.extra_ops.repeat(self.b_o_b,x_t.shape[0],axis=0))
		c_tilda_t = self.activation(T.dot(x_t, self.W_c_b) + T.dot(h_tm1, self.U_c_b) + T.extra_ops.repeat(self.b_c_b,x_t.shape[0],axis=0))

		c_t = f_t * c_tm1 + i_t * c_tilda_t
		h_t = o_t * self.activation(c_t)

		return h_t,c_t

	def output(self):
		X = self.layer_below.output()
		[forward_pass, cells], ups = theano.scan(fn=self.recurrence,
					sequences=[X],
					outputs_info=[T.extra_ops.repeat(self.h0,X.shape[1],axis=0), T.extra_ops.repeat(self.c0,X.shape[1],axis=0)],
					n_steps=X.shape[0],
					truncate_gradient=self.truncate_gradient
				)
		[backward_pass, cells], ups = theano.scan(fn=self.recurrence_backward,
					sequences=[X],
					outputs_info=[T.extra_ops.repeat(self.h0_b,X.shape[1],axis=0), T.extra_ops.repeat(self.c0_b,X.shape[1],axis=0)],
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

