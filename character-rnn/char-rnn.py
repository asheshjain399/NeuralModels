import numpy as np
import theano
from theano import tensor as T
from generateTrainDataonText import createTrain
from neuralmodels.utils import permute, load
from neuralmodels.costs import softmax_loss
from neuralmodels.models import RNN
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete
from neuralmodels.layers import softmax, simpleRNN, OneHot, LSTM

def text_prediction(class_ids_reverse,p_labels):
	N = p_labels.shape[1]
	T = p_labels.shape[0]
	text_output = []
	for i in range(N):
		t = ''
		for j in p_labels[:,i]:
			t = t + class_ids_reverse[j]
		text_output.append(t)
	return text_output

if __name__ == '__main__':
	num_samples = 10000
	num_validation = 100
	num_train = num_samples - num_validation
	len_samples = 300

	epochs = 30
	batch_size = 100
	learning_rate_decay = 0.97
	decay_after=5


	[X,Y,num_classes,class_ids_reverse] = createTrain('shakespeare_input.txt',num_samples,len_samples)
	inputD = num_classes
	outputD = num_classes

	permutation = permute(num_samples)
	X = X[:,permutation]
	Y = Y[:,permutation]
	X_tr = X[:,:num_train]
	Y_tr = Y[:,:num_train]
	X_valid = X[:,num_train:]
	Y_valid = Y[:,num_train:]
	
	# Creating network layers
	layers = [OneHot(num_classes),LSTM(),LSTM(),LSTM(activation_str='tanh',activation_gate='sigmoid',init='orthogonal',truncate_gradient=50,size=128,weights=None,seq_output=False),softmax(num_classes)]

	#trY = T.lmatrix()

	trY = T.lvector()

	# Initializing network
	rnn = RNN(layers,softmax_loss,trY,1e-3)

	# Fitting model
	rnn.fitModel(X_tr,Y_tr[-1],1,'checkpoints/',epochs,batch_size,learning_rate_decay,decay_after)

	# Printing a generated sentence	
	out = rnn.predict_language_model(X_valid[:,:1],1000,OutputSampleFromDiscrete)
	
	# Print the sentence here
	text_produced =  text_prediction(class_ids_reverse,out)
