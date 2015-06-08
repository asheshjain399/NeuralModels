import theano
from theano import tensor as T

def softmax_loss(p_t,y):
	shape = p_t.shape
	is_tensor3 = p_t.ndim > 2
	t = 1
	if is_tensor3:
		t = shape[0]
		p_t = p_t.reshape((shape[0]*shape[1],shape[2]))
		y = y.flatten()

	return - t*(T.mean(T.log(p_t)[T.arange(y.shape[0]), y]))		
