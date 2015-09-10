from headers import *

class multilayerLSTM(object):
	def __init__(self,layers,skip_input=False,skip_output=False,weights=None,nested_layers=True):
		self.settings = locals()
		del self.settings['self']
		self.nested_layers = nested_layers
		self.layers = layers
		self.skip_input = skip_input
		self.skip_output = skip_output
		self.size = 0	
		self.weights = weights

	def connect(self,layer_below):
		self.layer_below = layer_below
		self.layers[0].connect(self.layer_below)
		self.inputD = self.layer_below.size
		if self.skip_input:
			for i in range(1,len(self.layers)):
				self.layers[i].skip_input = True
				self.layers[i].connect(self.layers[i-1],self.layer_below)
		else:
			for i in range(1,len(self.layers)):
				self.layers[i].connect(self.layers[i-1])
		if self.skip_output:
			for i in range(len(self.layers)):
				self.size += self.layers[i].size
		else:
			self.size = self.layers[-1].size

		self.params = []
		for l in self.layers:
			self.params.extend(l.params)
			
	def output(self):
		if not self.skip_output:
			return self.layers[-1].output()
		else:
			concatenate_output = []
			for l in self.layers:
				concatenate_output.append(l.output())
			return T.concatenate(concatenate_output,axis=2)
