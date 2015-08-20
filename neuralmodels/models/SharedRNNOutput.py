from headers import *

class SharedRNNOutput(object):
	def __init__(self,shared_layers,layer_1,layer_2,layer_1_detection,layer_1_anticipation,layer_2_detection,layer_2_anticipation,cost,Y_1_detection,Y_2_detection,Y_1_anticipation,Y_2_anticipation,learning_rate,update_type=RMSprop()):
		self.settings = locals()
		del self.settings['self']
		self.shared_layers = shared_layers
		self.layer_1 = layer_1
		self.layer_2 = layer_2
		self.layer_1_anticipation = layer_1_anticipation
		self.layer_1_detection = layer_1_detection
		self.layer_2_anticipation = layer_2_anticipation
		self.layer_2_detection = layer_2_detection
		self.L2_sqr = theano.shared(value=np.float32(0.0))
		for i in range(1, len(self.shared_layers)):
			self.shared_layers[i].connect(self.shared_layers[i-1])
			self.L2_sqr += self.shared_layers[i].L2_sqr  
		self.layer_1[0].connect(self.shared_layers[-1])
		self.layer_2[0].connect(self.shared_layers[-1])
		for i in range(1, len(self.layer_1)):
			self.layer_1[i].connect(self.layer_1[i-1])
			self.L2_sqr += self.layer_1[i].L2_sqr  
		for i in range(1, len(self.layer_2)):
			self.layer_2[i].connect(self.layer_2[i-1])
			self.L2_sqr += self.layer_2[i].L2_sqr  
		self.layer_1_detection[0].connect(self.layer_1[-1])
		self.layer_1_anticipation[0].connect(self.layer_1[-1])
		self.layer_2_detection[0].connect(self.layer_2[-1])
		self.layer_2_anticipation[0].connect(self.layer_2[-1])


		self.X = self.shared_layers[0].input
		self.X_1 = self.layer_1[0].input
		self.X_2 = self.layer_2[0].input
	
		self.Y_pr_1_detection = self.layer_1_detection[0].output()
		self.Y_pr_1_anticipation = self.layer_1_anticipation[0].output()
		self.Y_1_detection = Y_1_detection
		self.Y_1_anticipation = Y_1_anticipation
		
		self.Y_pr_2_detection = self.layer_2_detection[0].output()
		self.Y_pr_2_anticipation = self.layer_2_anticipation[0].output()
		self.Y_2_detection = Y_2_detection
		self.Y_2_anticipation = Y_2_anticipation

		self.cost_layer_1_detection = cost(self.Y_pr_1_detection,self.Y_1_detection)
		self.cost_layer_1_anticipation = cost(self.Y_pr_1_anticipation,self.Y_1_anticipation)
		self.cost_layer_1 = self.cost_layer_1_detection + self.cost_layer_1_anticipation

		self.cost_layer_2_detection = cost(self.Y_pr_2_detection,self.Y_2_detection)
		self.cost_layer_2_anticipation = cost(self.Y_pr_2_anticipation,self.Y_2_anticipation)
		self.cost_layer_2 = self.cost_layer_2_detection + self.cost_layer_2_anticipation

	        self.params_shared = []
		for l in self.shared_layers:
	                if hasattr(l,'params'):
				self.params_shared.extend(l.params)

	        self.params_layer_1 = []
		for l in self.layer_1:
	                if hasattr(l,'params'):
				self.params_layer_1.extend(l.params)
		for l in self.shared_layers:
	                if hasattr(l,'params'):
				self.params_layer_1.extend(l.params)
		self.params_layer_1.extend(self.layer_1_detection[0].params)
		self.params_layer_1.extend(self.layer_1_anticipation[0].params)


	        self.params_layer_2 = []
		for l in self.layer_2:
	                if hasattr(l,'params'):
				self.params_layer_2.extend(l.params)
		for l in self.shared_layers:
	                if hasattr(l,'params'):
				self.params_layer_2.extend(l.params)
		self.params_layer_2.extend(self.layer_2_detection[0].params)
		self.params_layer_2.extend(self.layer_2_anticipation[0].params)

		
		#rmsprop = RMSprop()
		self.updates_layer_1 = update_type.get_updates(self.params_layer_1,self.cost_layer_1)
		self.updates_layer_2 = update_type.get_updates(self.params_layer_2,self.cost_layer_2)

		self.train_layer_1 = theano.function([self.X,self.X_1,self.Y_1_detection,self.Y_1_anticipation],self.cost_layer_1,updates=self.updates_layer_1)
		self.train_layer_2 = theano.function([self.X,self.X_2,self.Y_2_detection,self.Y_2_anticipation],self.cost_layer_2,updates=self.updates_layer_2)
		self.predict_layer_1_detection = theano.function([self.X,self.X_1],self.layer_1_detection[0].output())
		self.predict_layer_1_anticipation = theano.function([self.X,self.X_1],self.layer_1_anticipation[0].output())
		self.predict_layer_2_detection = theano.function([self.X,self.X_2],self.layer_2_detection[0].output())
		self.predict_layer_2_anticipation = theano.function([self.X,self.X_2],self.layer_2_anticipation[0].output())

	def fitModel(self,trX_shared_1,trX_1,trY_1_detection,trY_1_anticipation,trX_shared_2,trX_2,trY_2_detection,trY_2_anticipation,snapshot_rate=1,path=None,epochs=30,batch_size=50,learning_rate_decay=0.97,decay_after=10):
		from neuralmodels.loadcheckpoint import saveSharedRNNOutput
		X_shared_1_minibatch=[]
		X_1_minibatch=[]
		Y_1_minibatch_detection=[]
		Y_1_minibatch_anticipation=[]
		X_shared_2_minibatch=[]
		X_2_minibatch=[]
		Y_2_minibatch_detection=[]
		Y_2_minibatch_anticipation=[]
		num_examples = trX_1.shape[1]
		batches_in_one_epoch = int(num_examples / batch_size)
		
		loss_values = []
		for epoch in range(epochs):
			t0 = time.time()
			'''
			permutation = permute(num_examples)
			if self.X.ndim > 2:
				trX_shared_1 = trX_shared_1[:,permutation,:]
				trX_shared_2 = trX_shared_2[:,permutation,:]
				trX_1 = trX_1[:,permutation,:]
				trX_2 = trX_2[:,permutation,:]
			else:
				trX_shared_1 = trX_shared_1[:,permutation]
				trX_shared_2 = trX_shared_2[:,permutation]
				trX_1 = trX_1[:,permutation]
				trX_2 = trX_2[:,permutation]
			
			if self.Y.ndim > 1:
				trY_1 = trY_1[:,permutation]
				trY_2 = trY_2[:,permutation]
			else:
				trY_1 = trY_1[permutation]
				trY_2 = trY_2[permutation]
			'''
			for j in range(batches_in_one_epoch):
				'''
				if self.X.ndim > 2:
					X_shared_1_minibatch = trX_shared_1[:,j*batch_size:(j+1)*batch_size,:]
					X_shared_2_minibatch = trX_shared_2[:,j*batch_size:(j+1)*batch_size,:]
					X_1_minibatch = trX_1[:,j*batch_size:(j+1)*batch_size,:]
					X_2_minibatch = trX_2[:,j*batch_size:(j+1)*batch_size,:]
				else:
					X_shared_1_minibatch = trX_shared_1[:,j*batch_size:(j+1)*batch_size]
					X_shared_2_minibatch = trX_shared_2[:,j*batch_size:(j+1)*batch_size]
					X_1_minibatch = trX_1[:,j*batch_size:(j+1)*batch_size]
					X_2_minibatch = trX_2[:,j*batch_size:(j+1)*batch_size]
				if self.Y.ndim > 1:
					Y_1_minibatch = trY_1[:,j*batch_size:(j+1)*batch_size]
					Y_2_minibatch = trY_2[:,j*batch_size:(j+1)*batch_size]
				else:
					Y_1_minibatch = trY_1[j*batch_size:(j+1)*batch_size]
					Y_2_minibatch = trY_2[j*batch_size:(j+1)*batch_size]
				'''

				X_shared_1_minibatch = trX_shared_1
				X_shared_2_minibatch = trX_shared_2
				X_1_minibatch = trX_1
				X_2_minibatch = trX_2
				Y_1_minibatch_detection = trY_1_detection
				Y_1_minibatch_anticipation = trY_1_anticipation
				Y_2_minibatch_detection = trY_2_detection
				Y_2_minibatch_anticipation = trY_2_anticipation

				loss_layer_1 = self.train_layer_1(X_shared_1_minibatch,X_1_minibatch,Y_1_minibatch_detection,Y_1_minibatch_anticipation)
				loss_layer_2 = self.train_layer_2(X_shared_2_minibatch,X_2_minibatch,Y_2_minibatch_detection,Y_2_minibatch_anticipation)
				total_loss = loss_layer_1 + loss_layer_2
				loss_values.append(total_loss)
				print "epoch={0} loss_1={1} loss_2={2} total={3}".format(epoch,loss_layer_1,loss_layer_2,total_loss)
				
			if path and epoch % snapshot_rate == 0:
				print 'saving snapshot checkpoint.{0}'.format(epoch)
				saveSharedRNNOutput(self,"{0}checkpoint.{1}".format(path,epoch))
				f = open('{0}logfile'.format(path),'w')
				for v in loss_values:
					f.write('{0}\n'.format(v))
				f.close()
			t1 = time.time()
			print 'Epoch took {0} seconds'.format(t1-t0)

			#if epoch > decay_after:
			#	self.learning_rate *= learning_rate_decay 


	def predict_output(self,teX_shared,teX,predictfn,layer=1,output_layer='detection'):
		prediction = []
		if layer == 1:
			if output_layer == 'detection':
				prediction = self.predict_layer_1_detection(teX_shared,teX)
			else:
				prediction = self.predict_layer_1_anticipation(teX_shared,teX)
		else:
			if output_layer == 'detection':
				prediction = self.predict_layer_2_detection(teX_shared,teX)
			else:
				prediction = self.predict_layer_2_anticipation(teX_shared,teX)

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
