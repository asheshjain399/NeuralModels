import theano
from theano import tensor as T
import numpy as np

def softmax_loss(p_t,y):
	shape = p_t.shape
	is_tensor3 = p_t.ndim > 2
	t = 1
	if is_tensor3:
		t = shape[0]
		p_t = p_t.reshape((shape[0]*shape[1],shape[2]))
		y = y.flatten()

	return - t*(T.mean(T.log(p_t)[T.arange(y.shape[0]), y]))

def softmax_decay_loss(p_t,y):

	shape = p_t.shape

	def recurrence(x_t,y_t,log_loss,t):
		log_loss_new = log_loss + T.exp(-t)*T.sum(T.log(x_t)[T.arange(y_t.shape[0]), y_t])
		t_new = t - 1
		return log_loss_new, t_new

	[log_loss_list, cells], ups = theano.scan(fn=recurrence,
					sequences=[p_t, y],
					outputs_info=[theano.shared(value=0.0), p_t.shape[0]-1],
					n_steps=p_t.shape[0]
				)
	return - (1.0/shape[1])*log_loss_list[-1]
	'''

	def recurrence(x_t, y_t, t):
		h_t = T.exp(-t)*x_t
		t_new = t - 1
		return h_t, t_new

	[p_t_new, cells], ups = theano.scan(fn=recurrence,
					sequences=[p_t, y],
					outputs_info=[None, p_t.shape[0]-1],
					n_steps=p_t.shape[0]
				)

	p_t_new = p_t_new.reshape((shape[0]*shape[1],shape[2]))
	y = y.flatten()
	'''


	#return - shape[0]*(T.mean(T.log(p_t_new)[T.arange(y.shape[0]), y]))
