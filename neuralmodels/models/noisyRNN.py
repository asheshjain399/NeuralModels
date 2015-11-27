from headers import *

class noisyRNN(object):
	def __init__(self,layers,cost,Y,learning_rate,update_type=RMSprop(),clipnorm=0.0,weight_decay=0.0):
		self.settings = locals()
		del self.settings['self']
		self.layers = layers
		self.learning_rate = learning_rate
		self.clipnorm = clipnorm	
		self.std = T.scalar(dtype=theano.config.floatX)
		self.weight_decay = weight_decay
		
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

		self.cost =  cost(self.Y_pr,self.Y) + self.weight_decay * layers[-1].L2_sqr
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

		print 'Number of parameters ',self.num_params
			
		[self.updates,self.grads] = self.update_type.get_updates(self.params,self.cost)
		self.train = theano.function([self.X,self.Y,self.learning_rate,self.std],self.cost,updates=self.updates,on_unused_input='warn')
		self.predict = theano.function([self.X,self.std],self.layers[-1].output(),on_unused_input='warn')
		self.prediction_loss = theano.function([self.X,self.Y,self.std],self.cost,on_unused_input='warn')

		self.norm = T.sqrt(sum([T.sum(g**2) for g in self.grads]))
		self.grad_norm = theano.function([self.X,self.Y,self.std],self.norm,on_unused_input='warn')

	def fitModel(self,trX,trY,snapshot_rate=1,path=None,epochs=30,batch_size=50,learning_rate=1e-3,
		learning_rate_decay=0.97,std=1e-5,decay_after=-1,trX_validation=None,trY_validation=None,
		trX_forecasting=None,trY_forecasting=None,rng=np.random.RandomState(1234567890),iter_start=None,
		decay_type=None,decay_schedule=None,decay_rate_schedule=None,
		use_noise=False,noise_schedule=None,noise_rate_schedule=None,maxiter=10000,poseDataset=None,unNormalizeData=None):

		from neuralmodels.loadcheckpoint import save

		'''Saving first 5 training examples. This is for visualization of training error'''	
		training_trajectories = trX[:,:5,:]
		if path:
			fname = 'train_example'
			self.saveForecastedMotion(training_trajectories,path,fname)

		'''While calculating the loos we ignore these many initial time steps'''
		delta_t_ignore = 0 #50
	
		'''If loading an existing model then some of the parameters needs to be restored'''
		epoch_count = 0
		validation_set = []
		loss_after_each_minibatch = []
		complete_logger = ''
		iterations = 0
		if iter_start > 0:
			if path:
				lines = open('{0}logfile'.format(path)).readlines()
				for i in range(iter_start):
					line = lines[i]
					values = line.strip().split(',')
					print values
					if len(values) == 1:
						loss_after_each_minibatch.append(float(values[0]))
						validation_set.append(-1)
					elif len(values) == 2:
						loss_after_each_minibatch.append(float(values[0]))
						validation_set.append(float(values[1]))
				#if os.path.exists('{0}complete_log'.format(path)):
				#	complete_logger = open('{0}complete_log'.format(path)).read()
				#	complete_logger = complete_logger[:epoch_count]
			iterations = iter_start + 1

		N = trX.shape[1]
		outputDim = trY.ndim
		seq_length = trY.shape[0] - delta_t_ignore
		feature_dim = trY.shape[2]
		batches_in_one_epoch = int(np.ceil(N*1.0 / batch_size))
		numrange = np.arange(N)
		X = []
		Y = []
		
		#iterations = epoch_count * batches_in_one_epoch * 1.0
	
		'''Comverting validation set to a single array when doing drop joint experiments'''
		gth = None
		T1 = -1
		N1 = -1	
		if poseDataset.drop_features and unNormalizeData is not None:
			[T1,N1,D1] = trY_validation.shape
			trY_validation_new = np.zeros((T1,N1,poseDataset.data_mean.shape[0]))
			for i in range(N1):
				trY_validation_new[:,i,:] = np.float32(unNormalizeData(trY_validation[:,i,:],poseDataset.data_mean,poseDataset.data_std,poseDataset.dimensions_to_ignore))
			gth = trY_validation_new[poseDataset.drop_start-1:poseDataset.drop_end-1,:,poseDataset.drop_id]


		Tvalidation = 0
		Dvalidation = 0
		if (trX_validation is not None):
			Tvalidation = trX_validation.shape[0] - delta_t_ignore
			Dvalidation = trX_validation.shape[2]
		epoch = 0
		print 'batches in one epoch ',batches_in_one_epoch
		#for epoch in range(epoch_count,epochs):
		while iterations <= maxiter:
			t0 = time.time()

			'''Learning rate decay.'''	
			if decay_type:
				if decay_type == 'continuous' and decay_after > 0 and epoch > decay_after:
					learning_rate *= learning_rate_decay
				elif decay_type == 'schedule' and decay_schedule is not None:
					for i in range(len(decay_schedule)):
						if decay_schedule[i] > 0 and iterations > decay_schedule[i]:
							learning_rate *= decay_rate_schedule[i]
							decay_schedule[i] = -1

			'''Set noise level.'''	
			if use_noise and noise_schedule is not None:
				for i in range(len(noise_schedule)):
					if noise_schedule[i] > 0 and iterations >= noise_schedule[i]:
						std = noise_rate_schedule[i]
						noise_schedule[i] = -1

			if poseDataset is not None:
				trX,trY = poseDataset.getMalikFeatures(std)
				trX_validation,trY_validation = poseDataset.getMalikValidationFeatures(std)
				trX_forecasting,trY_forecasting = poseDataset.getMalikTrajectoryForecasting(std)

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
				g = self.grad_norm(X,Y,std)
				loss_after_each_minibatch.append(loss)
				validation_set.append(-1)
				iterations += 1

				termout = 'e={1} iter={8} m={2} lr={5} g_l2={4} noise={7} loss={0} normalized={3} skel_err={6}'.format(loss,epoch,j,(loss*1.0/(seq_length*feature_dim)),g,learning_rate,np.sqrt(loss*1.0/seq_length),std,iterations)
				#termout = 'e={1} m={2} lr={5} g_l2={4} noise={7} loss={0} normalized={3} skel_err={6}'.format(loss,epoch,j,(loss*1.0/(feature_dim)),g,learning_rate,np.sqrt(loss*1.0),std)
				complete_logger += termout + '\n'
				print termout
		
				'''Trajectory forecasting on validation set'''
				if (trX_forecasting is not None) and (trY_forecasting is not None) and path and int(iterations) % snapshot_rate == 0:
					forecasted_motion = self.predict_sequence(trX_forecasting,sequence_length=trY_forecasting.shape[0])
					fname = 'forecast_iteration_{0}'.format(iterations)
					self.saveForecastedMotion(forecasted_motion,path,fname)

					skel_err = np.mean(np.sqrt(np.sum(np.square((forecasted_motion - trY_forecasting)),axis=2)),axis=1)
					err_per_dof = skel_err / trY_forecasting.shape[2]
					fname = 'forecast_error_iteration_{0}'.format(iterations)
					self.saveForecastError(skel_err,err_per_dof,path,fname)

				'''Saving the learned model so far'''
				if path and int(iterations) % snapshot_rate == 0:
					print 'saving snapshot checkpoint.{0}'.format(iterations)
					save(self,"{0}checkpoint.{1}".format(path,iterations))

			'''Computing error on validation set'''
			if (trX_validation is not None) and (trY_validation is not None) and (not poseDataset.drop_features):
				validation_error = self.prediction_loss(trX_validation,trY_validation,std)
				validation_set[-1] = validation_error
				termout = 'Validation: loss={0} normalized={1} skel_err={2}'.format(validation_error,(validation_error*1.0/(Tvalidation*Dvalidation)),np.sqrt(validation_error*1.0/Tvalidation))
				complete_logger += termout + '\n'
				print termout
	
			if (trX_validation is not None) and (trY_validation is not None) and (poseDataset.drop_features) and (unNormalizeData is not None):
				prediction = self.predict_nextstep(trX_validation)
				prediction_new = np.zeros((T1,N1,poseDataset.data_mean.shape[0]))
				for i in range(N1):
					prediction_new[:,i,:] = np.float32(unNormalizeData(prediction[:,i,:],poseDataset.data_mean,poseDataset.data_std,poseDataset.dimensions_to_ignore))
				predict = prediction_new[poseDataset.drop_start-1:poseDataset.drop_end-1,:,poseDataset.drop_id]
				joint_error = np.linalg.norm(predict - gth)
				validation_set[-1] = joint_error
				termout = 'Missing joint error {0}'.format(joint_error )
				complete_logger += termout + '\n'
				print termout

			if path:
				print 'Dir: ',path

				'''Writing training error and validation error in a log file'''
				f = open('{0}logfile'.format(path),'w')
				for l,v in zip(loss_after_each_minibatch,validation_set):
					f.write('{0},{1}\n'.format(l,v))
				f.close()
				f = open('{0}complete_log'.format(path),'w')
				f.write(complete_logger)
				f.close()

				'''Get error on training trajectories'''
				#training_prediction = self.predict(training_trajectories,1e-5)
				#fname = 'train_error_epoch_{0}'.format(epoch)
				#self.saveForecastedMotion(training_prediction,path,fname)

			t1 = time.time()
			termout = 'Epoch took {0} seconds'.format(t1-t0)
			complete_logger += termout + '\n'
			print termout
			epoch += 1

	def saveForecastError(self,skel_err,err_per_dof,path,fname):
		f = open('{0}{1}'.format(path,fname),'w')
		for i in range(skel_err.shape[0]):
			f.write('T={0} {1}, {2}\n'.format(i,skel_err[i],err_per_dof[i]))
		f.close()

	def saveForecastedMotion(self,forecast,path,fname):
		T = forecast.shape[0]
		N = forecast.shape[1]
		D = forecast.shape[2]
		for j in range(N):
			motion = forecast[:,j,:]
			f = open('{0}{2}_N_{1}'.format(path,j,fname),'w')
			for i in range(T):
				st = ''
				for k in range(D):
					st += str(motion[i,k]) + ','
				st = st[:-1]
				f.write(st+'\n')
			f.close()
		
	'''Predicts future movements'''	
	def predict_sequence(self,teX_original,sequence_length=100):
		teX = copy.deepcopy(teX_original)
		future_sequence = []
		for i in range(sequence_length):
			prediction = self.predict(teX,1e-5)
			prediction = prediction[-1]
			teX = np.append(teX,[prediction],axis=0)
			future_sequence.append(prediction)
		del teX
		return np.array(future_sequence)

	def predict_nextstep(self,teX):
		prediction = self.predict(teX,1e-5)
		return prediction

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
