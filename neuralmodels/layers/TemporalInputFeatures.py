from headers import *

class TemporalInputFeatures(object):
	'''
	Use this layer to input dense features for RNN
	dim = Time x Num_examples x Feature_dimension
	'''
	def __init__(self,size,weights=None,skip_input=False,jump_up=False):
		self.settings = locals()
		del self.settings['self']
		self.size=size
		self.input=T.tensor3(dtype=theano.config.floatX)
		self.inputD=size
		self.params=[]
		self.weights=weights
		self.L2_sqr = theano.shared(value=np.float32(0.0))
		self.skip_input = skip_input
		self.jump_up = jump_up

	def output(self,seq_output=True):
		return self.input

