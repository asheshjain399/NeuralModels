# This optimization module is taken from Passage: https://github.com/IndicoDataSolutions/Passage/tree/master/passage
import theano
import theano.tensor as T
import numpy as np


def clip_norm(g, c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g*c/n, g)
    return g

def clip_norms(gs, c):
    norm = T.sqrt(sum([T.sum(g**2) for g in gs]))
    return [clip_norm(g, c, norm) for g in gs]

class Regularizer(object):

    def __init__(self, l1=0., l2=0., maxnorm=0.):
        self.__dict__.update(locals())

    def max_norm(self, p, maxnorm):
        if maxnorm > 0:
            norms = T.sqrt(T.sum(T.sqr(p), axis=0))
            desired = T.clip(norms, 0, maxnorm)
            p = p * (desired/ (1e-7 + norms))
        return p

    def gradient_regularize(self, p, g):
        g += p * self.l2
        g += T.sgn(p) * self.l1
        return g

    def weight_regularize(self, p):
        p = self.max_norm(p, self.maxnorm)
        return p


class Update(object):

    def __init__(self, regularizer=Regularizer(), clipnorm=0.):
        self.__dict__.update(locals())

    def get_updates(self, params, grads):
        raise NotImplementedError



class RMSprop(Update):

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            acc = theano.shared(p.get_value() * 0.)
            acc_new = self.rho * acc + (1 - self.rho) * g ** 2
            updates.append((acc, acc_new))

            updated_p = p - self.lr * (g / T.sqrt(acc_new + self.epsilon))
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates

