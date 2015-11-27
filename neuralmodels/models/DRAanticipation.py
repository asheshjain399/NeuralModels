from headers import *

class DRAanticipation(object):
	def __init__(self,edgeRNNs,nodeRNNs,outputLayer,nodeToEdgeConnections,edgeListComplete,cost,nodeLabels,learning_rate,clipnorm=0.0,update_type=RMSprop(),weight_decay=0.0,train_for='detection'):
		'''
		edgeRNNs and nodeRNNs are dictionary with keys as RNN name and value is a list of layers
		
		nodeToEdgeConnections is a dictionary with keys as nodeRNNs name and value is another dictionary whose keys are edgeRNNs the nodeRNN is connected to and value is a list of size-2 which indicates the features to choose from the unConcatenateLayer 

		nodeLabels is a dictionary with keys as node names and values as Theano matrix
		'''
		self.settings = locals()
		del self.settings['self']
		
		self.edgeRNNs = edgeRNNs
		self.nodeRNNs = nodeRNNs
		self.nodeToEdgeConnections = nodeToEdgeConnections
		self.edgeListComplete = edgeListComplete
		self.nodeLabels = nodeLabels
		self.learning_rate = learning_rate
		self.clipnorm = clipnorm
		self.weight_decay = weight_decay
		self.outputLayer = outputLayer
		nodeTypes = nodeRNNs.keys()
		edgeTypes = edgeRNNs.keys()

		self.train_for = train_for
		self.cost = {}
		self.X = {}
		self.Y_pr = {}
		self.Y_pr_last_timestep = {}
		self.Y = {}
		self.params = {}
		self.updates = {}
		self.train_node = {}
		self.predict_node = {}
		self.predict_node_last_timestep = {}
		self.masterlayer = {}
		self.grads = {}
		self.predict_node_loss = {}
		self.grad_norm = {}
		self.norm = {}

		self.update_type = update_type
		self.update_type.lr = self.learning_rate
		self.update_type.clipnorm = self.clipnorm

		self.std = T.scalar(dtype=theano.config.floatX)

		for et in edgeTypes:
			layers = self.edgeRNNs[et]
			for i in range(1,len(layers)):
				layers[i].connect(layers[i-1])
				if layers[i].__class__.__name__ == 'AddNoiseToInput':
					layers[i].std = self.std
		
		for nt in nodeTypes:
			self.params[nt] = []
			self.masterlayer[nt] = unConcatenateVectors(nodeToEdgeConnections[nt])
			self.X[nt] = self.masterlayer[nt].input

			'''We first connect all the edgeRNNs (building the network bottom-up)'''
			nodeLayers = self.nodeRNNs[nt]
			edgesConnectedTo = nodeToEdgeConnections[nt].keys()
			layers_below = []
			for et in edgeListComplete:
				if et not in edgesConnectedTo:
					continue
				edgeLayers = self.edgeRNNs[et]
				layers_below.append(edgeLayers)
				edgeLayers[0].input = self.masterlayer[nt].output(et)
				for l in edgeLayers:
					if hasattr(l,'params'):
						self.params[nt].extend(l.params)

			'''We now connect the bottom layer of nodeRNN with the concatenated output of edgeRNNs'''
			cv = ConcatenateVectors()
			cv.connect(layers_below)
			nodeLayers[0].connect(cv)
			
			'''Finally we connect the layers of NodeRNN'''
			for i in range(1,len(nodeLayers)):
				nodeLayers[i].connect(nodeLayers[i-1])

			for l in nodeLayers:
				if hasattr(l,'params'):
					self.params[nt].extend(l.params)

			for l in self.outputLayer[nt]:
				l.connect(nodeLayers[-1])
				if hasattr(l,'params'):
					self.params[nt].extend(l.params)

			if len(self.outputLayer[nt]) == 1:

				self.Y_pr[nt] = self.outputLayer[nt][0].output()
				#self.Y_pr[nt] = nodeLayers[-1].output()
				self.Y[nt] = self.nodeLabels[nt]
				
				self.cost[nt] = cost(self.Y_pr[nt],self.Y[nt]) 
			
				[self.updates[nt],self.grads[nt]] = self.update_type.get_updates(self.params[nt],self.cost[nt])
			
				self.train_node[nt] = theano.function([self.X[nt],self.Y[nt],self.learning_rate,self.std],self.cost[nt],updates=self.updates[nt],on_unused_input='ignore')
		
				self.predict_node[nt] = {}
	
				self.predict_node[nt][train_for] = theano.function([self.X[nt],self.std],self.Y_pr[nt],on_unused_input='ignore')
		
				#self.predict_node_loss[nt] = theano.function([self.X[nt],self.Y[nt],self.std],self.cost[nt],on_unused_input='ignore')
			
				self.norm[nt] = T.sqrt(sum([T.sum(g**2) for g in self.grads[nt]]))
			
				self.grad_norm[nt] = theano.function([self.X[nt],self.Y[nt],self.std],self.norm[nt],on_unused_input='ignore')
		
			if len(self.outputLayer[nt]) == 2:
				
				self.Y_pr[nt] = {}
				self.Y_pr[nt]['detection'] = self.outputLayer[nt][0].output()
				self.Y_pr[nt]['anticipation'] = self.outputLayer[nt][1].output()
				#self.Y_pr[nt] = nodeLayers[-1].output()

				self.Y[nt] = {}
				self.Y[nt]['detection'] = self.nodeLabels[nt]['detection']
				self.Y[nt]['anticipation'] = self.nodeLabels[nt]['anticipation']
				
				self.cost[nt] = cost(self.Y_pr[nt]['detection'],self.Y[nt]['detection']) + cost(self.Y_pr[nt]['anticipation'],self.Y[nt]['anticipation'])
			
				[self.updates[nt],self.grads[nt]] = self.update_type.get_updates(self.params[nt],self.cost[nt])
			
				self.train_node[nt] = theano.function([self.X[nt],self.Y[nt]['detection'],self.Y[nt]['anticipation'],self.learning_rate,self.std],self.cost[nt],updates=self.updates[nt],on_unused_input='ignore')
				self.predict_node[nt] = {}

				self.predict_node[nt]['detection'] = theano.function([self.X[nt],self.std],self.Y_pr[nt]['detection'],on_unused_input='ignore')
				self.predict_node[nt]['anticipation'] = theano.function([self.X[nt],self.std],self.Y_pr[nt]['anticipation'],on_unused_input='ignore')
	
				#self.predict_node_loss[nt] = theano.function([self.X[nt],self.Y[nt],self.std],self.cost[nt],on_unused_input='ignore')
			
				self.norm[nt] = T.sqrt(sum([T.sum(g**2) for g in self.grads[nt]]))
			
				self.grad_norm[nt] = theano.function([self.X[nt],self.Y[nt]['detection'],self.Y[nt]['anticipation'],self.std],self.norm[nt],on_unused_input='ignore')

		self.num_params = 0
		for nt in nodeTypes:
			nodeLayers = self.nodeRNNs[nt]
			for layer in nodeLayers:
				if hasattr(layer,'params'):
					for par in layer.params:
						val = par.get_value()
						temp = 1
						for i in range(val.ndim):
							temp *= val.shape[i]		
						self.num_params += temp
			for layer in self.outputLayer[nt]:
				if hasattr(layer,'params'):
					for par in layer.params:
						val = par.get_value()
						temp = 1
						for i in range(val.ndim):
							temp *= val.shape[i]		
						self.num_params += temp

		for et in edgeTypes:
			edgeLayers = self.edgeRNNs[et]
			for layer in edgeLayers:
				if hasattr(layer,'params'):
					for par in layer.params:
						val = par.get_value()
						temp = 1
						for i in range(val.ndim):
							temp *= val.shape[i]		
						self.num_params += temp
		print 'Number of parameters in DRA: ',self.num_params

	def fitModel(self,trX,trY,snapshot_rate=1,path=None,epochs=30,batch_size=50,learning_rate=1e-3,
		learning_rate_decay=0.97,std=1e-5,decay_after=-1,trX_validation=None,trY_validation=None,
		trX_forecasting=None,trY_forecasting=None,rng=np.random.RandomState(1234567890),iter_start=None,
		decay_type=None,decay_schedule=None,decay_rate_schedule=None,
		use_noise=False,noise_schedule=None,noise_rate_schedule=None,
		new_idx=None,featureRange=None,poseDataset=None,graph=None,maxiter=10000,predictfn=None,train_for='detection'):
	
		from neuralmodels.loadcheckpoint import saveDRA

		'''If loading an existing model then some of the parameters needs to be restored'''
		epoch_count = 0
		iterations = 0
		validation_set = []
		skel_loss_after_each_minibatch = []
		loss_after_each_minibatch = []
		complete_logger = ''
		if iter_start > 0:
			if path:
				lines = open('{0}logfile'.format(path)).readlines()
				for i in range(iter_start):
					line = lines[i]
					values = line.strip().split(',')
					print values
					if len(values) == 1:
						skel_loss_after_each_minibatch.append(float(values[0]))
						validation_set.append(-1)
					elif len(values) == 2:
						skel_loss_after_each_minibatch.append(float(values[0]))
						validation_set.append(float(values[1]))
				#if os.path.exists('{0}complete_log'.format(path)):
				#	complete_logger = open('{0}complete_log'.format(path)).read()
				#	complete_logger = complete_logger[:epoch_count]
			iterations = iter_start + 1

		tr_X = {}
		tr_Y = {}
		Nmax = 0
		outputDim = 0
		unequalSize = False
		numExamples = {}
		seq_length = 0
		skel_dim = 0
		

		nodeTypes = self.nodeRNNs.keys()
		print "nodeTypes: ",nodeTypes
		for nt in nodeTypes:
			tr_X[nt] = []
			tr_Y[nt] = []

		nodeNames = trX.keys()
		for nm in nodeNames:
			N = trX[nm].shape[1]
			seq_length = trX[nm].shape[0]
			outputDim = trY[nm]['detection'].ndim
			if outputDim > 2:
				skel_dim += trY[nm]['detection'].shape[2]

			numExamples[nm] = N
			if Nmax == 0:
				Nmax = N
			if not Nmax == N:
				if N > Nmax:
					Nmax = N
				unequalSize = True
				

		if unequalSize:
			batch_size = Nmax
		
		batches_in_one_epoch = 1
		for nm in nodeNames:
			N = trX[nm].shape[1]
			batches_in_one_epoch = int(np.ceil(N*1.0 / batch_size))
			break

		print "batches in each epoch ",batches_in_one_epoch
		print nodeNames	
		#iterations = epoch_count * batches_in_one_epoch * 1.0
		numrange = np.arange(Nmax)
		#for epoch in range(epoch_count,epochs):
		validation_file = None
		if path is not None:
			if train_for == 'joint':
				validation_file = open('{0}{2}_{1}'.format(path,'joint_validation_acc_detection',train_for),'w')
				validation_file.close()
				validation_file = open('{0}{2}_{1}'.format(path,'joint_validation_acc_anticipation',train_for),'w')
				validation_file.close()
			else:
				validation_file = open('{0}{2}_{1}'.format(path,'validation_acc',train_for),'w')
				validation_file.close()


		epoch = 0
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

			'''Permuting before mini-batch iteration'''
			if not unequalSize:
				shuffle_list = rng.permutation(numrange)
				for nm in nodeNames:
					trX[nm] = trX[nm][:,shuffle_list,:]
					if outputDim == 2:
						trY[nm] = trY[nm][:,shuffle_list]
					elif outputDim == 3:
						trY[nm] = trY[nm][:,shuffle_list,:]

			for j in range(batches_in_one_epoch):

				examples_taken_from_node = 0	
				for nm in nodeNames:
					nt = nm.split(':')[1]
					
					tr_X[nt] = trX[nm]
					tr_Y[nt] = {}
					tr_Y[nt]['detection'] = trY[nm]['detection']
					tr_Y[nt]['anticipation'] = trY[nm]['anticipation']
			
					'''
					if(len(tr_X[nt])) == 0:
						examples_taken_from_node = min((j+1)*batch_size,numExamples[nm]) - j*batch_size
						tr_X[nt] = copy.deepcopy(trX[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm]),:])
						if outputDim == 2:
							tr_Y[nt] = copy.deepcopy(trY[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm])])
						elif outputDim == 3:
							tr_Y[nt] = copy.deepcopy(trY[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm]),:])
					else:
						tr_X[nt] = np.concatenate((tr_X[nt],trX[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm]),:]),axis=1)
						if outputDim == 2:
							tr_Y[nt] = np.concatenate((tr_Y[nt],trY[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm])]),axis=1)
						elif outputDim == 3:
							tr_Y[nt] = np.concatenate((tr_Y[nt],trY[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm]),:]),axis=1)
					'''

				loss = 0.0
				skel_loss = 0.0
				grad_norms = []
				losses = {}
				for nt in nodeTypes:
					loss_for_current_node = 0.0
					g = []
					if train_for == 'joint':
						loss_for_current_node = self.train_node[nt](tr_X[nt],tr_Y[nt]['detection'],tr_Y[nt]['anticipation'],learning_rate,std)
						g = self.grad_norm[nt](tr_X[nt],tr_Y[nt]['detection'],tr_Y[nt]['anticipation'],std)
					else:
						loss_for_current_node = self.train_node[nt](tr_X[nt],tr_Y[nt][train_for],learning_rate,std)
						g = self.grad_norm[nt](tr_X[nt],tr_Y[nt][train_for],std)
					losses[nt] = loss_for_current_node
					grad_norms.append(g)
					loss += loss_for_current_node
				iterations += 1
				loss_after_each_minibatch.append(loss)
				validation_set.append(-1)
				termout = 'e={1} iter={8} m={2} lr={5} g_l2={4} noise={7} loss={0} H={3} O={6}'.format(loss,epoch,j,(losses['H']*1.0/(seq_length)),grad_norms,learning_rate,(losses['O']*1.0/seq_length),std,iterations)
				complete_logger += termout + '\n'
				print termout
			
				del tr_X
				del tr_Y
							
				tr_X = {}
				tr_Y = {}
				for nt in nodeTypes:
					tr_X[nt] = []
					tr_Y[nt] = []

				if int(iterations) % snapshot_rate == 0:
					print 'saving snapshot checkpoint.{0}'.format(int(iterations))
					saveDRA(self,"{0}checkpoint.{1}".format(path,int(iterations)))
		

				'''Computing error on validation set'''
				if (trX_validation is not None) and (trY_validation is not None) and (predictfn is not None) and (int(iterations) % snapshot_rate == 0):
					if train_for == 'joint':
						predict_detection = self.predict_output(trX_validation,trY_validation['detection'],predictfn,'detection')
						predict_anticipation = self.predict_output(trX_validation,trY_validation['anticipation'],predictfn,'anticipation')

						[detection_belief,detection_labels] = self.predict_nextstep(trX_validation,trY_validation['detection'],predictfn,'detection')
						[anticipation_belief,anticipation_labels] = self.predict_nextstep(trX_validation,trY_validation['anticipation'],predictfn,'anticipation')


						validation_acc_detection = {}
						validation_pr_detection = {}
						validation_re_detection = {}
						for nm in predict_detection.keys():
							validation_acc_detection[nm] = self.microaccuracy(predict_detection[nm])
							temp = self.confusionMat(predict_detection[nm])
							validation_pr_detection[nm] = temp[-2]
							validation_re_detection[nm] = temp[-1]
						validation_acc_anticipation = {}
						validation_pr_anticipation = {}
						validation_re_anticipation = {}
						for nm in predict_anticipation.keys():
							validation_acc_anticipation[nm] = self.microaccuracy(predict_anticipation[nm])
							temp = self.confusionMat(predict_anticipation[nm])
							validation_pr_anticipation[nm] = temp[-2]
							validation_re_anticipation[nm] = temp[-1]
						termout = 'Detection Validation: H={0} O={1}'.format(validation_acc_detection['H:H'],validation_acc_detection['O:O'])
						complete_logger += termout + '\n'
						print termout
						termout = 'Anticipation Validation: H={0} O={1}'.format(validation_acc_anticipation['H:H'],validation_acc_anticipation['O:O'])
						complete_logger += termout + '\n'
						print termout

						validation_file = None
						if path is not None:
							validation_file = open('{0}{2}_{1}'.format(path,'joint_validation_acc_detection',train_for),'a')
							validation_file.write('iter={0} H={1} [{3};{4}] O={2} [{5};{6}]\n'.format(iterations,validation_acc_detection['H:H'],validation_acc_detection['O:O'],validation_pr_detection['H:H'],validation_re_detection['H:H'],validation_pr_detection['O:O'],validation_re_detection['O:O']))
							validation_file.close()
						if path is not None:
							validation_file = open('{0}{2}_{1}'.format(path,'joint_validation_acc_anticipation',train_for),'a')
							validation_file.write('iter={0} H={1} [{3};{4}] O={2} [{5};{6}]\n'.format(iterations,validation_acc_anticipation['H:H'],validation_acc_anticipation['O:O'],validation_pr_anticipation['H:H'],validation_re_anticipation['H:H'],validation_pr_anticipation['O:O'],validation_re_anticipation['O:O']))
							validation_file.close()
						
						if path is not None:
							cc = 1
							for b,ypr,y_ in zip(detection_belief['H:H'],detection_labels['H:H'],trY_validation['detection']['H:H']):
								belief_file = open('{0}detection_belief_{1}_{2}'.format(path,cc,iterations),'w')
								for i in range(b.shape[0]):
									st = '{0} {1} '.format(y_[i,0],ypr[i,0])
									for val in b[i,:]:
										st = st + '{0} '.format(val)
									st = st.strip() + '\n'
									belief_file.write(st)
								belief_file.close()
								cc += 1

							cc = 1
							for b,ypr,y_ in zip(anticipation_belief['H:H'],anticipation_labels['H:H'],trY_validation['anticipation']['H:H']):
								belief_file = open('{0}anticipation_belief_{1}_{2}'.format(path,cc,iterations),'w')
								for i in range(b.shape[0]):
									st = '{0} {1} '.format(y_[i,0],ypr[i,0])
									for val in b[i,:]:
										st = st + '{0} '.format(val)
									st = st.strip() + '\n'
									belief_file.write(st)
								belief_file.close()
								cc += 1

							cc = 1
							for b,ypr,y_ in zip(detection_belief['O:O'],detection_labels['O:O'],trY_validation['detection']['O:O']):
								belief_file = open('{0}detection_objbelief_{1}_{2}'.format(path,cc,iterations),'w')
								for i in range(b.shape[0]):
									st = '{0} {1} '.format(y_[i,0],ypr[i,0])
									for val in b[i,:]:
										st = st + '{0} '.format(val)
									st = st.strip() + '\n'
									belief_file.write(st)
								belief_file.close()
								cc += 1

							cc = 1
							for b,ypr,y_ in zip(anticipation_belief['O:O'],anticipation_labels['O:O'],trY_validation['anticipation']['O:O']):
								belief_file = open('{0}anticipation_objbelief_{1}_{2}'.format(path,cc,iterations),'w')
								for i in range(b.shape[0]):
									st = '{0} {1} '.format(y_[i,0],ypr[i,0])
									for val in b[i,:]:
										st = st + '{0} '.format(val)
									st = st.strip() + '\n'
									belief_file.write(st)
								belief_file.close()
								cc += 1


					else:
						predict = self.predict_output(trX_validation,trY_validation[train_for],predictfn,train_for=train_for)
						validation_acc = {}
						validation_pr = {}
						validation_re = {}
						for nm in predict.keys():
							validation_acc[nm] = self.microaccuracy(predict[nm])
							temp = self.confusionMat(predict[nm])
							validation_pr[nm] = temp[-2]
							validation_re[nm] = temp[-1]
							
						termout = 'Validation: H={0} O={1}'.format(validation_acc['H:H'],validation_acc['O:O'])
						complete_logger += termout + '\n'
						validation_file = None
						if path is not None:
							validation_file = open('{0}{2}_{1}'.format(path,'validation_acc',train_for),'a')
							#validation_file.write('iter={0} H={1} O={2}\n'.format(iterations,validation_acc['H:H'],validation_acc['O:O']))
							validation_file.write('iter={0} H={1} [{3};{4}] O={2} [{5};{6}]\n'.format(iterations,validation_acc['H:H'],validation_acc['O:O'],validation_pr['H:H'],validation_re['H:H'],validation_pr['O:O'],validation_re['O:O']))
							validation_file.close()
						print termout

			'''Saving the learned model so far'''
			if path:
				
				print 'Dir: ',path				
				'''Writing training error and validation error in a log file'''
				f = open('{0}logfile'.format(path),'w')
				for l,v in zip(skel_loss_after_each_minibatch,validation_set):
					f.write('{0},{1}\n'.format(l,v))
				f.close()
				f = open('{0}complete_log'.format(path),'w')
				f.write(complete_logger)
				f.close()
			

			t1 = time.time()
			termout = 'Epoch took {0} seconds'.format(t1-t0)
			complete_logger += termout + '\n'
			print termout
			epoch += 1

	def microaccuracy(self,predict):
		Y_pr = predict['prediction']
		Y = predict['gt']
		pr = np.array(Y) - np.array(Y_pr)
		wrng_pr = np.where(pr == 0)
		acc = 1.0*(len(wrng_pr[0])) / len(Y_pr)
		return acc

	def predict_nextstep(self,teX,teY,predictfn,train_for='detection'):
		nodeNames = teX.keys()

		predict = {}
		labels = {}
		for nm in nodeNames:
			predict[nm] = []
			labels[nm] = []
			nt = nm.split(':')[1]
			
			for X,Y in zip(teX[nm],teY[nm]):
				prediction = self.predict_node[nt][train_for](X,1e-5)
				predict[nm].append(prediction[:,0,:])
				shape = prediction.shape
				Y_pr = []
				if prediction.ndim > 2:
					# prediction dim = T x N x D
					# Sequence prediction
					Y_pr = prediction.reshape(shape[0]*shape[1],shape[2])
					Y_pr = predictfn(Y_pr)
					Y_pr = Y_pr.reshape(shape[0],shape[1])
					# Output dim = T x N
				else:
					# prediction dim = N x D
					# Single prediction at the end of sequence
					Y_pr = predictfn(prediction)
					# Output dim = N
				labels[nm].append(Y_pr)
		return predict,labels
			


	def predict_output(self,teX,teY,predictfn,train_for='detection'):
		nodeNames = teX.keys()
		
		predict = {}
		for nm in nodeNames:
			predict[nm] = {}
			predict[nm]['gt'] = []
			predict[nm]['prediction'] = []

			nt = nm.split(':')[1]
			for X,Y in zip(teX[nm],teY[nm]):
				prediction = self.predict_node[nt][train_for](X,1e-5)
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
				predict[nm]['gt'].extend(list(Y[:,0]))
				predict[nm]['prediction'].extend(list(prediction[:,0]))
		return predict

	def confusionMat(self,predict):

		P = np.array(predict['prediction'])
		Y = np.array(predict['gt'])

		Y = Y - 1
		P = P - 1

		size = np.max(Y) + 1
		confMat = np.zeros((size,size))
		for p,y in zip(P,Y):
			if p < 0 or p >= size:
				continue 
			confMat[p,y] += 1.0
		col_sum = np.reshape(np.sum(confMat,axis=1),(size,1))
		row_sum = np.reshape(np.sum(confMat,axis=0),(1,size))
		precision_confMat = confMat/np.repeat(col_sum,size,axis=1)
		recall_confMat = confMat/np.repeat(row_sum,size,axis=0)

		pr = 0.0
		cc = 0.0
		for x in np.diag(precision_confMat):
			if not math.isnan(x):
				pr += x
				cc += 1.0
		if cc > 0:
			pr = pr / cc		

		re = 0.0
		cc = 0.0
		for x in np.diag(recall_confMat):
			if not math.isnan(x):
				re += x
				cc += 1.0
		if cc > 0:
			re = re / cc		

		return confMat,precision_confMat,recall_confMat,pr,re
