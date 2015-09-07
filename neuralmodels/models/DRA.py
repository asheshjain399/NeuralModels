from headers import *

class DRA(object):
	def __init__(self,edgeRNNs,nodeRNNs,nodeToEdgeConnections,cost,nodeLabels,learning_rate,clipnorm=0.0,update_type=RMSprop()):
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
		self.nodeLabels = nodeLabels
		self.update_type = update_type		
		self.learning_rate = learning_rate
		self.clipnorm = clipnorm

		nodeTypes = nodeRNNs.keys()
		edgeTypes = edgeRNNs.keys()

		self.cost = {}
		self.X = {}
		self.Y_pr = {}
		self.Y = {}
		self.params = {}
		self.updates = {}
		self.train_node = {}
		self.predict_node = {}
		self.masterlayer = {}

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

			nodeLayers = self.nodeRNNs[nt]
			for i in range(1,len(nodeLayers)):
				nodeLayers[i].connect(nodeLayers[i-1])

			for l in nodeLayers:
				if hasattr(l,'params'):
					self.params[nt].extend(l.params)

			edgesConnectedTo = nodeToEdgeConnections[nt].keys()
			layers_below = []
			for et in edgesConnectedTo:
				edgeLayers = self.edgeRNNs[et]
				layers_below.append(edgeLayers)
				edgeLayers[0].input = self.masterlayer[nt].output(et)
				for l in edgeLayers:
					if hasattr(l,'params'):
						self.params[nt].extend(l.params)

			cv = ConcatenateVectors()
			cv.connect(layers_below)
			nodeLayers[0].connect(cv)

			self.Y_pr[nt] = nodeLayers[-1].output()
			self.Y[nt] = self.nodeLabels[nt]
			self.cost[nt] = cost(self.Y_pr[nt],self.Y[nt])
			self.updates[nt] = self.update_type.get_updates(self.params[nt],self.cost[nt])
			self.train_node[nt] = theano.function([self.X[nt],self.Y[nt],self.learning_rate,self.std],self.cost[nt],updates=self.updates[nt],on_unused_input='warn')
			self.predict_node[nt] = theano.function([self.X[nt],self.std],self.Y_pr[nt],on_unused_input='warn')
		

	def fitModel(self,trX,trY,snapshot_rate=1,path=None,epochs=30,batch_size=50,learning_rate=1e-3,learning_rate_decay=0.97,std=1e-5,decay_after=-1,trX_validation=None,trY_validation=None,evaluateModel=None,rng=np.random.RandomState(1234567890)):
		from neuralmodels.loadcheckpoint import saveDRA
		
		tr_X = {}
		tr_Y = {}
		Nmax = 0
		outputDim = 0
		unequalSize = False
		numExamples = {}

		nodeTypes = self.nodeRNNs.keys()
		for nt in nodeTypes:
			tr_X[nt] = []
			tr_Y[nt] = []

		nodeNames = trX.keys()
		for nm in nodeNames:
			N = trX[nm].shape[1]
			outputDim = trY[nm].ndim
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
		numrange = np.arange(Nmax)
		loss_after_each_minibatch = []
		for epoch in range(epochs):
			t0 = time.time()

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
	
				for nm in nodeNames:
					nt = nm.split(':')[1]
					if(len(tr_X[nt])) == 0:
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

				loss = 0.0
				for nt in nodeTypes:
					loss += self.train_node[nt](tr_X[nt],tr_Y[nt],learning_rate,std)
				loss_after_each_minibatch.append(loss)
				print 'e={1} m={2} loss={0}'.format(loss,epoch,j)
				
				del tr_X
				del tr_Y
							
				tr_X = {}
				tr_Y = {}
				for nt in nodeTypes:
					tr_X[nt] = []
					tr_Y[nt] = []
			
			'''Learning rate decay'''	
			if decay_after > 0 and epoch > decay_after:
				learning_rate *= learning_rate_decay

			'''Saving the learned model so far'''
			if path and epoch % snapshot_rate == 0:
				print 'saving snapshot checkpoint.{0}'.format(epoch)
				saveDRA(self,"{0}checkpoint.{1}".format(path,epoch))
				f = open('{0}logfile'.format(path),'w')
				for v in loss_after_each_minibatch:
					f.write('{0}\n'.format(v))
				f.close()
		
			'''Computing error on validation set'''
			if (trX_validation is not None) and (trY_validation is not None) and (evaluateModel is not None):
				predict = self.predict_sequence(trX_validation)
				validation_error = evaluateModel(predict)
			
			t1 = time.time()
			print 'Epoch took {0} seconds'.format(t1-t0)


		def predict_output(self,teX):
			nodeNames = teX.keys()
			predict = {}
			for nm in nodeNames:
				nt = nm.split(':')[1]
				predict[nm] = self.predict_node[nt](teX[nm],1e-5)
			return predict

		def predict_sequence(self,teX,sequence_length=100):
			nodeNames = teX.keys()
			predict = {}
			for nm in nodeNames:
				predict[nm] = []
				nt = nm.split(':')[1]
				for i in range(sequence_length):
					prediction = self.predict_node[nt](teX[nm],1e-5)
					prediction = prediction[-1]
					teX[nm] = np.append(teX[nm],[prediction],axis=0)
					predict[nm].append(prediction)
				predict[nm] = np.array(predict[nm])
			return predict

