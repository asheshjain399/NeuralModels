from headers import *

'''

'''
class ConcatenateFeatures(object):
	def __init__(self,size,weights=None):
		self.settings = locals()
		del self.settings['self']
		self.size = size
		self.inputD = size 
		self.params=[]
		self.input=T.tensor3(dtype=theano.config.floatX)
		self.weights=weights
		self.L2_sqr = theano.shared(value=np.float32(0.0))

	def connect(self,layer_below):
		self.layer_below = layer_below
		self.size = self.size + self.layer_below.size

	def output(self):
		return T.concatenate([self.input, self.layer_below.output()], axis=2)

