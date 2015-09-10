from headers import *

class AddNoiseToInput(object):
	def __init__(self,weights=None,rng=None,skip_input=False,jump_up=False):
		self.settings = locals()
		del self.settings['self']
		self.rng = rng
		self.weights = weights
		self.params = []
		self.std=T.scalar(dtype=theano.config.floatX)
		self.skip_input = skip_input
		self.jump_up = jump_up
		if rng is None:
			self.rng = np.random
		self.theano_rng = T.shared_randomstreams.RandomStreams(self.rng.randint(2 ** 30))

	def connect(self,layer_below):
		self.layer_below = layer_below
		self.size = self.layer_below.size
		self.inputD = self.size
	
	def output(self):
		X = self.layer_below.output()
		out = T.switch(T.le(self.std,theano.shared(value=0.0)),X,(X + self.theano_rng.normal(size=X.shape,std=self.std,dtype=theano.config.floatX)))
		return out
