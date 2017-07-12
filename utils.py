import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layers import flatten

def KL_div(mu, sigma):
    '''KL divergence between N(mu,sigma**2) and N(0,1)'''
    return .5 * (mu**2 + sigma**2 - 2 * tf.log(sigma) - 1)

def KL_div2(mu, sigma, mu1, sigma1):
    '''KL divergence between N(mu,sigma**2) and N(mu1,sigma1**2)'''
    return 0.5 * ((sigma/sigma1)**2 + (mu - mu1)**2/sigma1**2 - 1 + 2*(tf.log(sigma1) - tf.log(sigma)))

def sample_lognormal(mean, sigma=None, sigma0=1.):
    '''Samples a log-normal using the reparametrization trick'''
    e = tf.random_normal(tf.shape(mean), mean=0., stddev=1.)
    return tf.exp(mean + sigma * sigma0 * e)

def spatial_global_mean(network):
    '''Averages features map along all spatial dimensions'''
    return tf.reduce_mean(network, [1,2])

def batch_average(x):
    '''Sum over all dimensions and averages over the first'''
    return tf.reduce_mean(tf.reduce_sum(flatten(x),1))