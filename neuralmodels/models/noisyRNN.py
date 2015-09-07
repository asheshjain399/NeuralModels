from headers import *

class noisyRNN(object):
	def __init__(self,layers,cost,Y,learning_rate,update_type=RMSprop(),clipnorm=0.0):
		self.settings = locals()
		del self.settings['self']
		self.layers = layers
		self.learning_rate = learning_rate
		self.clipnorm = clipnorm	
		self.std = T.scalar(dtype=theano.config.floatX)
		
		self.update_type = update_type
		self.update_type.lr = self.learning_rate
		self.update_type.clipnorm = self.clipnorm

		for i in range(1, len(layers)):
			layers[i].connect(layers[i-1])
			if layers[i].__class__.__name__ == 'AddNoiseToInput':
				layers[i].std = self.std

		self.X = layers[0].input
		self.Y_pr = layers[-1].output()
		self.Y = Y

		self.cost =  cost(self.Y_pr,self.Y)
	        self.params = []
		for l in self.layers:
	                if hasattr(l,'params'):
				self.params.extend(l.params)


		self.num_params = 0
		for par in self.params:
			val = par.get_value()
			temp = 1
			for i in range(val.ndim):
				temp *= val.shape[i]		
			self.num_params += temp
			
		self.updates = self.update_type.get_updates(self.params,self.cost)
		self.train = theano.function([self.X,self.Y,self.learning_rate,self.std],self.cost,updates=self.updates)
		self.predict = theano.function([self.X,self.std],self.layers[-1].output())
		self.prediction_loss = theano.function([self.X,self.Y,self.std],self.cost)
	
	def fitModel(self,trX,trY,snapshot_rate=1,path=None,epochs=30,batch_size=50,learning_rate=1e-3,
		learning_rate_decay=0.97,std=1e-5,decay_after=-1,trX_validation=None,trY_validation=None,
		trX_forecasting=None,trY_forecasting=None,rng=np.random.RandomState(1234567890)):

		from neuralmodels.loadcheckpoint import save

		N = trX.shape[1]
		outputDim = trY.ndim
		seq_length = trY.shape[0]
		feature_dim = trY.shape[2]
		batches_in_one_epoch = int(np.ceil(N*1.0 / batch_size))
		numrange = np.arange(N)
		loss_after_each_minibatch = []
		X = []
		Y = []
		
		Tvalidation = 0
		Dvalidation = 0
		if (trX_validation is not None):
			Tvalidation = trX_validation.shape[0]
			Dvalidation = trX_validation.shape[2]

		validation_set = []
		print 'batches in one epoch ',batches_in_one_epoch
		for epoch in range(epochs):
			t0 = time.time()

			'''Permuting before mini-batch iteration'''
			shuffle_list = rng.permutation(numrange)
			trX = trX[:,shuffle_list,:]
			if outputDim == 2:
				trY = trY[:,shuffle_list]
			elif outputDim == 3:
				trY = trY[:,shuffle_list,:]

			for j in range(batches_in_one_epoch):
				X = trX[:,j*batch_size:min((j+1)*batch_size,N),:]
				if outputDim == 2:
					Y = trY[:,j*batch_size:min((j+1)*batch_size,N)]
				elif outputDim == 3:
					Y = trY[:,j*batch_size:min((j+1)*batch_size,N),:]
				
				'''One iteration of training'''
				loss = self.train(X,Y,learning_rate,std)
				loss_after_each_minibatch.append(loss)
				validation_set.append(-1)
				print 'e={1} m={2} loss={0} normalized={3}'.format(loss,epoch,j,(loss*1.0/(seq_length*feature_dim)))

			'''Computing error on validation set'''
			if (trX_validation is not None) and (trY_validation is not None):
				validation_error = self.prediction_loss(trX_validation,trY_validation,1e-5)
				validation_set[-1] = validation_error
				print 'Validation: loss={0} normalized={1}'.format(validation_error,(validation_error*1.0/(Tvalidation*Dvalidation)))

			'''Trajectory forecasting'''
			if (trX_forecasting is not None) and (trY_forecasting is not None) and path:
				forecasted_motion = self.predict_sequence(trX_forecasting,sequence_length=trY_forecasting.shape[0])
				self.saveForecastedMotion(forecasted_motion,path,epoch)

			'''Learning rate decay'''	
			if decay_after > 0 and epoch > decay_after:
				learning_rate *= learning_rate_decay

			'''Saving the learned model so far'''
			if path and epoch % snapshot_rate == 0:
				print 'saving snapshot checkpoint.{0}'.format(epoch)
				save(self,"{0}checkpoint.{1}".format(path,epoch))

				'''Writing training error and validation error in a log file'''
				f = open('{0}logfile'.format(path),'w')
				for l,v in zip(loss_after_each_minibatch,validation_set):
					if v > 0:
						f.write('{0},{1}\n'.format(l,v))
					else:
						f.write('{0}\n'.format(l))
				f.close()

			
			t1 = time.time()
			print 'Epoch took {0} seconds'.format(t1-t0)

	def saveForecastedMotion(self,forecast,path,epoch):
		T = forecast.shape[0]
		N = forecast.shape[1]
		D = forecast.shape[2]
		for j in range(N):
			motion = forecast[:,j,:]
			f = open('{0}forecast_epoch_{2}_N_{1}'.format(path,j,epoch),'w')
			for i in range(T):
				st = ''
				for k in range(D):
					st += str(motion[i,k]) + ','
				st = st[:-1]
				f.write(st+'\n')
			f.close()
			
	def predict_sequence(self,teX,sequence_length=100):
		future_sequence = []
		for i in range(sequence_length):
			prediction = self.predict(teX,1e-5)
			prediction = prediction[-1]
			teX = np.append(teX,[prediction],axis=0)
			future_sequence.append(prediction)
		return np.array(future_sequence)

	def predict_output(self,teX,predictfn):
		prediction = self.predict(teX)
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
