#Layout of this package is inspired by Passage https://github.com/IndicoDataSolutions/Passage
import pdb
import time
import theano
import numpy as np
from theano import tensor as T
from utils import permute
from loadcheckpoint import save, saveSharedRNN, saveSharedRNNVectors, saveSharedRNNOutput, saveMultipleRNNsCombined
from updates import RMSprop, Adagrad
from layers.ConcatenateVectors import ConcatenateVectors

class RNN(object):
	def __init__(self,layers,cost,Y,learning_rate,update_type=RMSprop()):
		self.settings = locals()
		del self.settings['self']
		self.layers = layers
		self.L2_sqr = theano.shared(value=np.float32(0.0))
		for i in range(1, len(layers)):
			layers[i].connect(layers[i-1])
			self.L2_sqr += layers[i].L2_sqr  

		self.X = layers[0].input
		self.Y_pr = layers[-1].output()
		self.Y = Y

		self.cost = 0.0000*self.L2_sqr + cost(self.Y_pr,self.Y)
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
			
		#rmsprop = RMSprop()
		self.updates = update_type.get_updates(self.params,self.cost)

		self.train = theano.function([self.X,self.Y],self.cost,updates=self.updates)
		self.objective = theano.function([self.X,self.Y],self.cost)
		self.predict = theano.function([self.X],self.layers[-1].output())
	
	def fitModel(self,trX,trY,snapshot_rate=1,path=None,epochs=30,batch_size=50,learning_rate_decay=0.97,decay_after=10):
		X_minibatch=[]
		Y_minibatch=[]
		num_examples = trX.shape[1]
		batches_in_one_epoch = int(num_examples / batch_size)
		
		loss_values = []
		for epoch in range(epochs):
			t0 = time.time()
			permutation = permute(num_examples)
			if self.X.ndim > 2:
				trX = trX[:,permutation,:]
			else:
				trX = trX[:,permutation]
			
			if self.Y.ndim > 1:
				trY = trY[:,permutation]
			else:
				trY = trY[permutation]
			
			for j in range(batches_in_one_epoch):
				if self.X.ndim > 2:
					X_minibatch = trX[:,j*batch_size:(j+1)*batch_size,:]
				else:
					X_minibatch = trX[:,j*batch_size:(j+1)*batch_size]
				if self.Y.ndim > 1:
					Y_minibatch = trY[:,j*batch_size:(j+1)*batch_size]
				else:
					Y_minibatch = trY[j*batch_size:(j+1)*batch_size]

				loss = self.train(X_minibatch,Y_minibatch)
				loss_values.append(loss)
				print "epoch={0} loss={1}".format(epoch,loss)
				
			if path and epoch % snapshot_rate == 0:
				print 'saving snapshot checkpoint.{0}'.format(epoch)
				save(self,"{0}checkpoint.{1}".format(path,epoch))
				f = open('{0}logfile'.format(path),'w')
				for v in loss_values:
					f.write('{0}\n'.format(v))
				f.close()
			t1 = time.time()
			print 'Epoch took {0} seconds'.format(t1-t0)

			#if epoch > decay_after:
			#	self.learning_rate *= learning_rate_decay 


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

	def predict_language_model(self,teX,seq_length,predictfn,Temperature=0.7):
		# Currently only supported with language modeling
		predict = theano.function([self.X],self.layers[-1].output(Temperature))
		future_sequence = []
		for i in range(seq_length):
			prediction = predict(teX)
			prediction = prediction[-1]
			prediction = predictfn(prediction)
			teX = np.append(teX,[prediction],axis=0)
			future_sequence.append(prediction)
		return np.array(future_sequence)
		# dim = seq_length x N


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

class SharedRNN(object):
	def __init__(self,shared_layers,layer_1,layer_2,cost,Y_1,Y_2,learning_rate,update_type=RMSprop()):
		self.settings = locals()
		del self.settings['self']
		self.shared_layers = shared_layers
		self.layer_1 = layer_1
		self.layer_2 = layer_2
		self.L2_sqr = theano.shared(value=np.float32(0.0))
		for i in range(1, len(shared_layers)):
			shared_layers[i].connect(shared_layers[i-1])
			self.L2_sqr += shared_layers[i].L2_sqr  
		layer_1[0].connect(shared_layers[-1])
		layer_2[0].connect(shared_layers[-1])
		for i in range(1, len(layer_1)):
			layer_1[i].connect(layer_1[i-1])
			self.L2_sqr += layer_1[i].L2_sqr  
		for i in range(1, len(layer_2)):
			layer_2[i].connect(layer_2[i-1])
			self.L2_sqr += layer_2[i].L2_sqr  



		self.X = shared_layers[0].input
		self.X_1 = layer_1[0].input
		self.X_2 = layer_2[0].input
		self.Y_pr_1 = layer_1[-1].output()
		self.Y_pr_2 = layer_2[-1].output()
		self.Y_1 = Y_1
		self.Y_2 = Y_2

		self.cost_layer_1 = 0.0000*self.L2_sqr + cost(self.Y_pr_1,self.Y_1) 
		self.cost_layer_2 = cost(self.Y_pr_2,self.Y_2)

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

	        self.params_layer_2 = []
		for l in self.layer_2:
	                if hasattr(l,'params'):
				self.params_layer_2.extend(l.params)
		for l in self.shared_layers:
	                if hasattr(l,'params'):
				self.params_layer_2.extend(l.params)

		
		#rmsprop = RMSprop()
		self.updates_layer_1 = update_type.get_updates(self.params_layer_1,self.cost_layer_1)
		self.updates_layer_2 = update_type.get_updates(self.params_layer_2,self.cost_layer_2)

		self.train_layer_1 = theano.function([self.X,self.X_1,self.Y_1],self.cost_layer_1,updates=self.updates_layer_1)
		self.train_layer_2 = theano.function([self.X,self.X_2,self.Y_2],self.cost_layer_2,updates=self.updates_layer_2)
		self.predict_layer_1 = theano.function([self.X,self.X_1],self.layer_1[-1].output())
		self.predict_layer_2 = theano.function([self.X,self.X_2],self.layer_2[-1].output())

	def fitModel(self,trX_shared_1,trX_1,trY_1,trX_shared_2,trX_2,trY_2,snapshot_rate=1,path=None,epochs=30,batch_size=50,learning_rate_decay=0.97,decay_after=10):
		X_shared_1_minibatch=[]
		X_1_minibatch=[]
		Y_1_minibatch=[]
		X_shared_2_minibatch=[]
		X_2_minibatch=[]
		Y_2_minibatch=[]
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
				Y_1_minibatch = trY_1
				Y_2_minibatch = trY_2


				loss_layer_1 = self.train_layer_1(X_shared_1_minibatch,X_1_minibatch,Y_1_minibatch)
				loss_layer_2 = self.train_layer_2(X_shared_2_minibatch,X_2_minibatch,Y_2_minibatch)
				total_loss = loss_layer_1 + loss_layer_2
				loss_values.append(total_loss)
				print "epoch={0} loss_1={1} loss_2={2} total={3}".format(epoch,loss_layer_1,loss_layer_2,total_loss)
				
			if path and epoch % snapshot_rate == 0:
				print 'saving snapshot checkpoint.{0}'.format(epoch)
				saveSharedRNN(self,"{0}checkpoint.{1}".format(path,epoch))
				f = open('{0}logfile'.format(path),'w')
				for v in loss_values:
					f.write('{0}\n'.format(v))
				f.close()
			t1 = time.time()
			print 'Epoch took {0} seconds'.format(t1-t0)

			#if epoch > decay_after:
			#	self.learning_rate *= learning_rate_decay 


	def predict_output(self,teX_shared,teX,predictfn,layer=1):
		prediction = []
		if layer == 1:
			prediction = self.predict_layer_1(teX_shared,teX)
		else:
			prediction = self.predict_layer_2(teX_shared,teX)

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

class SharedRNNVectors(object):
	def __init__(self,shared_layers,layer_1,layer_2,layer_1_output,layer_2_output,cost,Y_1,Y_2,learning_rate,update_type=RMSprop()):
		self.settings = locals()
		del self.settings['self']
		self.shared_layers = shared_layers
		self.layer_1 = layer_1
		self.layer_2 = layer_2
		self.layer_1_output = layer_1_output
		self.layer_2_output = layer_2_output
		self.L2_sqr = theano.shared(value=np.float32(0.0))
		for i in range(1, len(shared_layers)):
			shared_layers[i].connect(shared_layers[i-1])
			self.L2_sqr += shared_layers[i].L2_sqr  
		for i in range(1, len(layer_1)):
			layer_1[i].connect(layer_1[i-1])
			self.L2_sqr += layer_1[i].L2_sqr  
		for i in range(1, len(layer_2)):
			layer_2[i].connect(layer_2[i-1])
			self.L2_sqr += layer_2[i].L2_sqr  
		self.layer_1_output[0].connect(self.shared_layers[-1],self.layer_1[-1])
		self.layer_2_output[0].connect(self.shared_layers[-1],self.layer_2[-1])
		for i in range(1, len(layer_1_output)):
			layer_1_output[i].connect(layer_1_output[i-1])
			self.L2_sqr += layer_1_output[i].L2_sqr  
		for i in range(1, len(layer_2_output)):
			layer_2_output[i].connect(layer_2_output[i-1])
			self.L2_sqr += layer_2_output[i].L2_sqr  



		self.X = shared_layers[0].input
		self.X_1 = layer_1[0].input
		self.X_2 = layer_2[0].input
		self.Y_pr_1 = layer_1_output[-1].output()
		self.Y_pr_2 = layer_2_output[-1].output()
		self.Y_1 = Y_1
		self.Y_2 = Y_2

		self.cost_layer_1 = 0.0000*self.L2_sqr + cost(self.Y_pr_1,self.Y_1) 
		self.cost_layer_2 = cost(self.Y_pr_2,self.Y_2)

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
		for l in self.layer_1_output:
	                if hasattr(l,'params'):
				self.params_layer_1.extend(l.params)

	        self.params_layer_2 = []
		for l in self.layer_2:
	                if hasattr(l,'params'):
				self.params_layer_2.extend(l.params)
		for l in self.shared_layers:
	                if hasattr(l,'params'):
				self.params_layer_2.extend(l.params)
		for l in self.layer_2_output:
	                if hasattr(l,'params'):
				self.params_layer_2.extend(l.params)

		
		#rmsprop = RMSprop()
		self.updates_layer_1 = update_type.get_updates(self.params_layer_1,self.cost_layer_1)
		self.updates_layer_2 = update_type.get_updates(self.params_layer_2,self.cost_layer_2)

		self.train_layer_1 = theano.function([self.X,self.X_1,self.Y_1],self.cost_layer_1,updates=self.updates_layer_1)
		self.train_layer_2 = theano.function([self.X,self.X_2,self.Y_2],self.cost_layer_2,updates=self.updates_layer_2)
		self.predict_layer_1 = theano.function([self.X,self.X_1],self.layer_1_output[-1].output())
		self.predict_layer_2 = theano.function([self.X,self.X_2],self.layer_2_output[-1].output())

	def fitModel(self,trX_shared_1,trX_1,trY_1,trX_shared_2,trX_2,trY_2,snapshot_rate=1,path=None,epochs=30,batch_size=50,learning_rate_decay=0.97,decay_after=10):
		X_shared_1_minibatch=[]
		X_1_minibatch=[]
		Y_1_minibatch=[]
		X_shared_2_minibatch=[]
		X_2_minibatch=[]
		Y_2_minibatch=[]
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
				Y_1_minibatch = trY_1
				Y_2_minibatch = trY_2


				loss_layer_1 = self.train_layer_1(X_shared_1_minibatch,X_1_minibatch,Y_1_minibatch)
				loss_layer_2 = self.train_layer_2(X_shared_2_minibatch,X_2_minibatch,Y_2_minibatch)
				total_loss = loss_layer_1 + loss_layer_2
				loss_values.append(total_loss)
				print "epoch={0} loss_1={1} loss_2={2} total={3}".format(epoch,loss_layer_1,loss_layer_2,total_loss)
				
			if path and epoch % snapshot_rate == 0:
				print 'saving snapshot checkpoint.{0}'.format(epoch)
				saveSharedRNNVectors(self,"{0}checkpoint.{1}".format(path,epoch))
				f = open('{0}logfile'.format(path),'w')
				for v in loss_values:
					f.write('{0}\n'.format(v))
				f.close()
			t1 = time.time()
			print 'Epoch took {0} seconds'.format(t1-t0)

			#if epoch > decay_after:
			#	self.learning_rate *= learning_rate_decay 


	def predict_output(self,teX_shared,teX,predictfn,layer=1):
		prediction = []
		if layer == 1:
			prediction = self.predict_layer_1(teX_shared,teX)
		else:
			prediction = self.predict_layer_2(teX_shared,teX)

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

