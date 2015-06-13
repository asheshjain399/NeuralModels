import numpy as np
import random

def sampleSubSequences(length,num_samples=1,min_len=1,max_len=10):
	max_len = min(max_len,length)
	min_len = min(min_len,max_len)
	sequence = []
	for i in range(num_samples):
		l = random.randint(min_len,max_len)
		start_idx = random.randint(0,length-l)
		end_idx = start_idx + l
		if not (start_idx, end_idx) in sequence: 
			sequence.append((start_idx, end_idx))

	return sequence
