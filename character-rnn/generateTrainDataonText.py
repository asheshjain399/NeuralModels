import numpy as np 
from collections import Counter
import sys


def sample_text(input_text,num_samples,len_samples,class_ids):
	overall_data = []
	train_data = []
	label_data = []
	for i in range(num_samples):
		if (i+1)*(len_samples+1) >= len(input_text):
			break
		s_t = int(i*len_samples)
		e_t = int((i+1)*len_samples + 1)
		sample_text = input_text[s_t:e_t]
		overall_data.append([class_ids[x] for x in sample_text])
	overall_data = np.array(overall_data,dtype=np.int64)
	train_data = overall_data[:,:-1]
	label_data = overall_data[:,1:]
	return train_data,label_data

def createTrain(filename,num_samples=1000,len_samples=300):
	input_text = open(filename).read()
	print 'Number of characters in document {0}'.format(len(input_text))
	counter = Counter(input_text)
	num_classes = len(counter.keys())
	Y = range(num_classes)

	class_ids = {}
	class_ids_reverse = {}
	count = 0
	for t in counter:
		class_ids[t] = count
		class_ids_reverse[count] = t
		count+=1
	
	[train_data,label_data] = sample_text(input_text,num_samples,len_samples,class_ids)
	train_data = np.transpose(train_data)
	label_data = np.transpose(label_data)
	# dim = T x N
	
	return train_data,label_data,num_classes,class_ids_reverse

if __name__=="__main__":
	filename = 'shakespeare_input.txt'
	createTrain(filename,100)
	 
