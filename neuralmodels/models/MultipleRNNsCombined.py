from headers import *

class MultipleRNNsCombined(object):
	'''
	This model concatenates high-level features from multiple RNNs and feeds the final feature vector into a combined_layer
	Input:
		rnn_layers: is a list of list. [[layers_1],[layers_2],...,[layers_N]] where layers_n is a RNN
		combined_layer: this is the final layer concatenating features from rnn_layers 
	BUG BUG BUG: Currently this only works for len(rnn_layers) == 2
	'''
	def __init__(self,rnn_layers,combined_layer,cost,Y,learning_rate,update_type=RMSprop()):
		self.settings = locals()
		del self.settings['self']
		self.rnn_layers = rnn_layers
		self.combined_layer = combined_layer
		self.L2_sqr = theano.shared(value=np.float32(0.0))
	
		for layers in self.rnn_layers:
			for i in range(1,len(layers)):
				layers[i].connect(layers[i-1])

		self.feature_concatenation = ConcatenateVectors()
		self.feature_concatenation.connect(self.rnn_layers)
		combined_layer[0].connect(self.feature_concatenation)

		for i in range(1,len(combined_layer)):
			combined_layer[i].connect(combined_layer[i-1])
		
		self.X = []
		for layers in self.rnn_layers:
			self.X.append(layers[0].input)
		self.Y = Y
		self.Y_pr = combined_layer[-1].output()


		self.cost = cost(self.Y_pr,self.Y)

		self.params = []
	
		for layers in self.rnn_layers:
			for l in layers:
				if hasattr(l,'params'):
					self.params.extend(l.params)
		for l in combined_layer:
			if hasattr(l,'params'):
				self.params.extend(l.params)

		self.num_params = 0
		for par in self.params:
			val = par.get_value()
			temp = 1
			for i in range(val.ndim):
				temp *= val.shape[i]		
			self.num_params += temp
		
		#rmsprop = RMSprop()
		self.updates = update_type.get_updates(self.params,self.cost)

		self.train = theano.function([self.X[0],self.X[1],self.Y],self.cost,updates=self.updates)
		self.predict = theano.function([self.X[0],self.X[1]],self.combined_layer[-1].output())

	def fitModel(self,trX,trY,snapshot_rate=1,path=None,epochs=30,batch_size=50,learning_rate_decay=0.97,decay_after=10):
		from neuralmodels.loadcheckpoint import saveMultipleRNNsCombined
		X = []
		Y = []
		num_examples = trX[0].shape[1]
		batches_in_one_epoch = int(num_examples / batch_size)
		
		loss_values = []
		for epoch in range(epochs):
			t0 = time.time()
			for j in range(batches_in_one_epoch):
				X = trX
				Y = trY
				loss = self.train(X[0],X[1],Y)
				loss_values.append(loss)
				print "epoch={0} loss={1}".format(epoch,loss)
				
			if path and epoch % snapshot_rate == 0:
				print 'saving snapshot checkpoint.{0}'.format(epoch)
				saveMultipleRNNsCombined(self,"{0}checkpoint.{1}".format(path,epoch))
				f = open('{0}logfile'.format(path),'w')
				for v in loss_values:
					f.write('{0}\n'.format(v))
				f.close()
			t1 = time.time()
			print 'Epoch took {0} seconds'.format(t1-t0)


	def predict_output(self,teX,predictfn):
		prediction = self.predict(teX[0],teX[1])
		shape = prediction.shape
		if prediction.ndim > 2:
			# prediction dim = T x N x D
			# Sequence prediction
			prediction = prediction.reshape(shape[0]*shape[1],shape[2])
			prediction = predictfn(prediction)
			prediction = prediction.reshape(shape[0],shape[1])
			# Output dim = T x N
		else:
			# prediction dim = N x D
			# Single prediction at the end of sequence
			prediction = predictfn(prediction)
			# Output dim = N
		return prediction


