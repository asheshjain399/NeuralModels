#Layout of this package is inspired by Passage https://github.com/IndicoDataSolutions/Passage

import theano
import numpy as np
from theano import tensor as T
from utils import permute, save
from updates import RMSprop

class RNN(object):
	def __init__(self,layers,cost,Y,learning_rate):
		self.settings = locals()
		del self.settings['self']
		self.layers = layers
		for i in range(1, len(layers)):
			layers[i].connect(layers[i-1])

		self.X = layers[0].input
		self.Y_pr = layers[-1].output()
		self.Y = Y

		self.cost = cost(self.Y_pr,self.Y)
	        self.params = []
		for l in self.layers:
	                if hasattr(l,'params'):
				self.params.extend(l.params)
		
		
		rmsprop = RMSprop()
		self.updates = rmsprop.get_updates(self.params,self.cost)

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
			permutation = permute(num_examples)
			if self.X.ndim > 2:
				trX = trX[:,permutation,:]
			else:
				trX = trX[:,permutation]
			trY = trY[:,permutation]
			
			for j in range(batches_in_one_epoch):
				if self.X.ndim > 2:
					X_minibatch = trX[:,j*batch_size:(j+1)*batch_size,:]
				else:
					X_minibatch = trX[:,j*batch_size:(j+1)*batch_size]
				Y_minibatch = trY[:,j*batch_size:(j+1)*batch_size]
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

			#if epoch > decay_after:
			#	self.learning_rate *= learning_rate_decay 


				
	
	def predict_sequence(self,teX,teY,predictfn):
		prediction = self.predict(teX)
		shape = prediction.shape
		prediction = prediction.reshape(shape[0]*shape[1],shape[2])
		prediction = predictfn(prediction)
		prediction = prediction.reshape(shape[0],shape[1])
		return prediction
		# dim = T x N
	
	def predict_last(self,teX,teY,predictfn):
		prediction = self.predict(teX)
		prediction = prediction[-1]
		prediction = predictfn(prediction)
		return prediction
		# dim = N

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
