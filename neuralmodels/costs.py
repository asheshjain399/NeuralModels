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
	T = shape[0]
	N = shape[1]
	D = shape[2]
	exp_decay = np.ones((T,N,D)).astype(theano.config.floatX)
	for i in range(T):
		exp_decay[i,:,:] = np.exp(-np.log(T-i))
	exp_decay_theano = theano.shared(value=exp_decay)
	p_t = p_t*exp_decay_theano

	t = shape[0]
	p_t = p_t.reshape((shape[0]*shape[1],shape[2]))
	y = y.flatten()

	return - t*(T.mean(T.log(p_t)[T.arange(y.shape[0]), y]))		
