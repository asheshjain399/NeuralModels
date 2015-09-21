from headers import *

'''
This layer concatenates high-level representations from 
top-layer of multiple deep networks into a single features vector.
'''

class ConcatenateVectors(object):
	def __init__(self,weights=None):
		self.settings = locals()
		del self.settings['self']
		self.params=[]
		self.weights=weights
		self.L2_sqr = theano.shared(value=np.float32(0.0))

	def connect(self,layers_below):
		self.size = 0
		self.layers_below = layers_below
		for layer in self.layers_below:
			self.size += layer[-1].size

	def output(self,seq_output=True):
		concatenate_output = []
		is_tensor3 = False

		for layer in self.layers_below:
			concatenate_output.append(layer[-1].output(seq_output=seq_output))
			if (layer[-1].output()).ndim > 2:
				is_tensor3 = True

		axis_concatenate = 1
		if is_tensor3:
			axis_concatenate = 2

		return T.concatenate(concatenate_output,axis=axis_concatenate)
