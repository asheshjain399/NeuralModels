from headers import *

class TemporalInputFeatures(object):
	'''
	Use this layer to input dense features for RNN
	dim = Time x Num_examples x Feature_dimension
	'''
	def __init__(self,size,weights=None):
		self.settings = locals()
		del self.settings['self']
		self.size=size
		self.input=T.tensor3(dtype=theano.config.floatX)
		self.inputD=size
		self.params=[]
		self.weights=weights
		self.L2_sqr = theano.shared(value=np.float32(0.0))
	def output(self):
		return self.input

