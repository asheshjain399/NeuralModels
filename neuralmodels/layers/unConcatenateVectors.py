from headers import *

class unConcatenateVectors(object):
	def __init__(self,idxValues,weights=None):
		self.settings = locals()
		del self.settings['self']
		self.input=T.tensor3(dtype=theano.config.floatX)
		self.idxValues = idxValues
		self.params=[]
		self.weights=weights
		self.L2_sqr = theano.shared(value=np.float32(0.0))
	def output(self,idx):
		low = self.idxValues[idx][0]
		high = self.idxValues[idx][1]
		return self.input[:,:,low:high]
