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

		nodeNames = nodeRNNs.keys()
		edgeNames = edgeRNNs.keys()

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

		for em in edgeNames:
			layers = self.edgeRNNs[em]
			for i in range(1,len(layers)):
				layers[i].connect(layers[i-1])
				if layers[i].__class__.__name__ == 'AddNoiseToInput':
					layers[i].std = self.std

		for nm in nodeNames:
			self.params[nm] = []
			self.masterlayer[nm] = unConcatenateVectors(nodeToEdgeConnections[nm])
			self.X[nm] = self.masterlayer[nm].input

			nodeLayers = self.nodeRNNs[nm]
			for i in range(1,len(nodeLayers)):
				nodeLayers[i].connect(nodeLayers[i-1])

			for l in nodeLayers:
				if hasattr(l,'params'):
					self.params[nm].extend(l.params)

			edgesConnectedTo = nodeToEdgeConnections[nm].keys()
			layers_below = []
			for em in edgesConnectedTo:
				edgeLayers = self.edgeRNNs[em]
				layers_below.append(edgeLayers)
				edgeLayers[0].input = self.masterlayer[nm].output(em)
				for l in edgeLayers:
					if hasattr(l,'params'):
						self.params[nm].extend(l.params)

			cv = ConcatenateVectors()
			cv.connect(layers_below)
			nodeLayers[0].connect(cv)

			self.Y_pr[nm] = nodeLayers[-1].output()
			self.Y[nm] = self.nodeLabels[nm]
			self.cost[nm] = cost(self.Y_pr[nm],self.Y[nm])
			self.updates[nm] = self.update_type.get_updates(self.params[nm],self.cost[nm])
			self.train_node[nm] = theano.function([self.X[nm],self.Y[nm],self.learning_rate,self.std],self.cost[nm],updates=self.updates[nm],on_unused_input='warn')
			self.predict_node[nm] = theano.function([self.X[nm],self.std],self.Y_pr[nm],on_unused_input='warn')
		

	def fitModel(self,trX,trY,snapshot_rate=1,path=None,epochs=30,batch_size=50,learning_rate=1e-3,learning_rate_decay=0.97,std=1e-5,decay_after=-1):
		from neuralmodels.loadcheckpoint import saveDRA
		loss_after_each_minibatch = []
		nodeNames = trX.keys()
		for epoch in range(epochs):
			t0 = time.time()
	
			# If you want to do minibatch then enter your code here
			loss_node = {}
			loss = 0.0
			for nm in nodeNames:
				loss_node[nm] = self.train_node[nm](trX[nm],trY[nm],learning_rate,std)
				loss += loss_node[nm]
			loss_after_each_minibatch.append(loss)
			print 'loss={0}'.format(loss)
			# End minibatch code here

			if decay_after > 0 and epoch > decay_after:
				learning_rate *= learning_rate_decay

			if path and epoch % snapshot_rate == 0:
				print 'saving snapshot checkpoint.{0}'.format(epoch)
				saveDRA(self,"{0}checkpoint.{1}".format(path,epoch))
		
			t1 = time.time()
			print 'Epoch took {0} seconds'.format(t1-t0)
