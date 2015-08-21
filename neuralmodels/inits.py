# This method is taken from Passage https://github.com/IndicoDataSolutions/Passage

import numpy as np

import theano
import theano.tensor as T

def uniform(shape, scale=0.05, rng=None):
	if rng is None:
		rng = np.random
	return theano.shared(value=rng.uniform(low=-scale,high=scale,size=shape).astype(theano.config.floatX))

def allones(shape, scale=0.05, rng=None):
	if rng is None:
		rng = np.random	
	return theano.shared(value=np.ones(shape).astype(theano.config.floatX))

def normal(shape, scale=0.05, rng=None):
	if rng is None:
		rng = np.random	
	return theano.shared(value=(rng.randn(*shape) * scale).astype(theano.config.floatX))

def orthogonal(shape, scale=1.1, rng=None):
	""" benanne lasagne ortho init (faster than qr approach)"""
	if rng is None:
		rng = np.random	
	flat_shape = (shape[0], np.prod(shape[1:]))
	a = rng.normal(0.0, 1.0, flat_shape)
	u, _, v = np.linalg.svd(a, full_matrices=False)
	q = u if u.shape == flat_shape else v # pick the one with the correct shape
	q = q.reshape(shape)
	return theano.shared(value=(scale * q[:shape[0], :shape[1]]).astype(theano.config.floatX))
