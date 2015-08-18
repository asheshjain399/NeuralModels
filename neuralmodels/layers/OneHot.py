from headers import *

def theano_one_hot(idx, n):
# This method is taken from 'Passage'
	z = T.zeros((idx.shape[0], n))
	one_hot = T.set_subtensor(z[T.arange(idx.shape[0]), idx], 1)
	return one_hot

class OneHot(object):
	def __init__(self,size,weights=None):
		self.settings = locals()
		del self.settings['self']
		self.size=size
		self.input=T.lmatrix()
		self.inputD=1
		self.params=[]
		self.weights=weights
		self.L2_sqr = theano.shared(value=np.float32(0.0))

	def output(self):
		return theano_one_hot(self.input.flatten(), self.size).reshape((self.input.shape[0], self.input.shape[1], self.size))

