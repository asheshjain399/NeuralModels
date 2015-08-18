from headers import *

class DRA(object):
	def __init__(self,edgeRNNs,nodeRNNs,nodeToEdgeConnections,cost,nodeLabels,learning_rate,update_type=RMSprop()):
		self.settings = locals()
		del self.settings['self']
		
		self.edgeRNNs = edgeRNNs
		self.nodeRNNs = nodeRNNs
		self.nodeToEdgeConnections = nodeToEdgeConnections
		self.nodeLabels = nodeLabels
		
		nodeNames = nodeRNNs.keys()
		edgeNames = edgeRNNs.keys()

		self.cost = {}
		self.X = {}
		self.Y_pr = {}
		self.Y = {}
		self.params = {}
		self.updates = {}
		self.masterlayer = {}


		for em in edgeNames:
			layers = self.edgeRNNs[em]
			for i in range(1,len(layers)):
				layers[i].connect(layers[i-1])

		for nm in nodeNames:
			self.params[nm] = []
			self.masterlayer[nm] = masterLayer(nodeToEdgeConnections[nm])
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
			self.updates[nm] = update_type.get_updates(self.params[nm],self.cost[nm])
			self.train_node[nm] = theano.function([self.X[nm],self.Y[nm]],self.cost[nm],updates=self.updates[nm])
			self.predict_node[nm] = theano.function([self.X[nm]],self.Y_pr[nm])

		'''
		self.predict_layer_1 = theano.function([self.X,self.X_1],self.layer_1[-1].output())
		self.predict_layer_2 = theano.function([self.X,self.X_2],self.layer_2[-1].output())
		'''

