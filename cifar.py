#!/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib.layers import flatten, linear, fully_connected, conv2d, batch_norm
from tensorflow.contrib.layers.python.layers import utils
from sacred import Experiment, Ingredient
from task import task_ingredient, Task
import numpy as np
from utils import *

ex = Experiment('cifar', ingredients=[task_ingredient])

img_h, img_w = 32, 32

@task_ingredient.config
def task_config():
    batch_size = 100
    learning_rate = 0.05
    drop1 = 80
    drop2 = 120
    drop3 = 160
    end_epoch = 200
    keep_prob = 0.5
    initial_keep_prob = 0.8
    optimizer = 'momentum'
    name = 'cifar'

@ex.config
def cfg():
    dropout = 'information'
    activations = 'relu'
    beta = 3.0
    max_alpha = 0.7
    lognorm_prior = False
    weight_decay = 0.001
    filter_percentage = 1.0
    alpha_mode = 'information'

@ex.named_config
def softplus():
    activations = 'softplus'
    lognorm_prior = True

class MyTask(Task):

    def __init__(self):
        train = np.load('datasets/cifar10-train.npz')
        test = np.load('datasets/cifar10-test.npz')
        self.dataset = {'train': (train['data'], train['labels']),
                        'test': (test['data'], test['labels'])}
        self.dataset['valid'] = self.dataset['test']

    @task_ingredient.capture
    def build_placeholders(self, batch_size):
        '''Creates the placeholders for this model'''
        self.keep_prob = tf.placeholder(tf.float32, shape=[]) 
        self.initial_keep_prob = tf.placeholder(tf.float32, shape=[]) 
        self.sigma0 = tf.placeholder(tf.float32, shape=[])
        self.x = tf.placeholder(tf.float32, shape=[batch_size,img_h,img_w,3])  # input (batch_size * x_size)
        self.y = tf.placeholder(tf.float32, shape=[batch_size,10]) 
        self.is_training = tf.placeholder(tf.bool, shape=[]) 

    @ex.capture
    def conv(self, inputs, num_outputs, activations, normalizer_fn = batch_norm, kernel_size=3, stride=1, scope=None):
        '''Creates a convolutional layer with default arguments'''
        if activations == 'relu':
            activation_fn = tf.nn.relu
        elif activations == 'softplus':
            activation_fn = tf.nn.softplus
        else:
            raise ValueError("Invalid activation function.")
        return conv2d( inputs = inputs,
            num_outputs = num_outputs,
            kernel_size = kernel_size,
            stride = stride,
            padding = 'SAME',
            activation_fn = activation_fn,
            normalizer_fn = normalizer_fn,
            normalizer_params = {'is_training' : self.is_training, 'updates_collections': None, 'decay': 0.9},
            scope=scope )

    @ex.capture
    def information_pool(self, inputs, max_alpha, alpha_mode, lognorm_prior, num_outputs=None, stride=2, scope=None):
        if num_outputs is None:
            num_ouputs = inputs.get_shape()[-1]
        # Creates the output convolutional layer
        network = self.conv(inputs, num_outputs=int(num_outputs), stride=stride)
        with tf.variable_scope(scope,'information_dropout'):
            # Computes the noise parameter alpha for the output
            alpha = conv2d(inputs, num_outputs=int(num_outputs), kernel_size=3,
                stride=stride, activation_fn=tf.sigmoid, scope='alpha')
            # Rescale alpha in the allowed range and add a small value for numerical stability
            alpha = 0.001 + max_alpha * alpha
            # Computes the KL divergence using either log-uniform or log-normal prior
            if not lognorm_prior:
                kl = - tf.log(alpha/(max_alpha + 0.001))
            else:
                mu1 = tf.get_variable('mu1', [], initializer=tf.constant_initializer(0.))
                sigma1 = tf.get_variable('sigma1', [], initializer=tf.constant_initializer(1.))
                kl = KL_div2(tf.log(tf.maximum(network,1e-4)), alpha, mu1, sigma1)
            tf.add_to_collection('kl_terms', kl)
        # Samples the noise with the given parameter
        e = sample_lognormal(mean=tf.zeros_like(network), sigma = alpha, sigma0 = self.sigma0)
        # Saves the log-output of the network (useful to compute the total correlation)
        tf.add_to_collection('log_network', tf.log(network * e))
        # Returns the noisy output of the dropout
        return network * e

    @ex.capture
    def conv_dropout(self, inputs, num_outputs, dropout):
        if dropout == 'information':
            network = self.information_pool(inputs, num_outputs=num_outputs)
        elif dropout == 'binary':
            network = self.conv(inputs, num_outputs, stride=2)
            network = tf.nn.dropout(network, self.keep_prob)
        elif dropout == 'none':
            network = self.conv(inputs, num_outputs, stride=2)
        else:
            raise ValueError("Invalid dropout value")
        return network

    @ex.capture
    def build_network(self, inputs, filter_percentage):
        network = inputs
        network = tf.nn.dropout(network, self.initial_keep_prob)
        # 32x32x3
        N = int(96*filter_percentage)
        print "Filter 1: %d" % N
        network = self.conv(network, N)
        network = self.conv(network, N)
        network = self.conv_dropout(network, N)
        # 16x16x96
        N = int(192*filter_percentage)
        print "Filter 2: %d" % N
        network = self.conv(network, N)
        network = self.conv(network, N)
        network = self.conv_dropout(network, N)
        # 8x8x192
        network = self.conv(network, N)
        network = self.representation = self.conv(network, N, kernel_size=1)
            
        network = self.conv(network, 10, kernel_size=1)
        network = spatial_global_mean(network)
        return network


    @ex.capture
    def build_loss(self, beta, task, weight_decay):
        batch_size = task['batch_size']
        with tf.variable_scope("network") as scope:
            network = self.build_network(self.x)
            logits = linear(network, num_outputs=10)
        with tf.name_scope('loss'):
            kl_terms = [ batch_average(kl) for kl in tf.get_collection('kl_terms') ]
            if not kl_terms:
                kl_terms = [ tf.constant(0.)]
            N_train = self.dataset['train'][0].shape[0]
            Lz = tf.add_n(kl_terms)/N_train
            Lx = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))
            beta = tf.constant(beta)
            L2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() ])
            loss = Lx + beta * Lz + weight_decay * L2
            correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(self.y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.loss = loss
        self.error = (1. - accuracy) * 100.
        self.Lx = Lx
        self.Lz = Lz
        self.beta = beta

    @task_ingredient.capture
    def train_batch(self, batch, stats, batch_size, keep_prob, initial_keep_prob):
        xtrain, ytrain = batch
        ytrain = np.eye(10)[ytrain]
        feed_dict = {self.x: xtrain, self.y: ytrain, self.sigma0: 1., self.keep_prob: keep_prob, self.initial_keep_prob: initial_keep_prob, self.learning_rate: self.current_learning_rate, self.is_training: True}
        batch_cost, batch_error, batch_Lx, batch_Lz, batch_beta, _ = self.sess.run( [ self.loss, self.error, self.Lx, self.Lz, self.beta, self.train_op], feed_dict)

        stats.push(Lx = batch_Lx)
        stats.push(Lz = batch_Lz)
        stats.push(train = batch_cost)
        stats.push(error = batch_error)
    
    # 
    @ex.capture
    def valid_batch(self, batch, stats):
        xtrain, ytrain = batch
        ytrain = np.eye(10)[ytrain]
        feed_dict = {self.x: xtrain, self.y: ytrain, self.sigma0: 0., self.keep_prob: 1., self.initial_keep_prob: 1.0, self.is_training: False}
        batch_cost, batch_error = self.sess.run( [ self.loss, self.error], feed_dict)

        stats.push(train = batch_cost)
        stats.push(error = batch_error)

    @ex.capture
    def dry_run(self):
        '''
        Since the statistics learned by batch normalization with dropout are
        incorrect when dropout is disabled,
        we do a dry run without dropout in order to relearn them before testing.
        '''
        for batch in self.iterate_minibatches('train'):
            xtrain, ytrain = batch
            ytrain = np.eye(10)[ytrain]
            feed_dict = {self.x: xtrain, self.y: ytrain, self.sigma0: 0., self.keep_prob: 1., self.initial_keep_prob: 1.0, self.is_training: True}
            batch_cost = self.sess.run( [ self.loss], feed_dict)

mytask = task = MyTask()

@ex.command
def train():
    task.initialize()
    task.train()

@ex.command
def test(load=True):
    if load:
        task.initialize(_load=True, _log_dir='valid/')
    print "Dry run..."
    task.dry_run()
    print "Validating..."
    task.valid()

@ex.command
def correlation(task,load=True):
    self = mytask
    if load:
        self.initialize(_load=True, _logging=False, _log_dir='other/')
    data = []
    for batch in self.iterate_minibatches('valid'):
        xtrain, ytrain = batch
        ytrain = np.eye(10)[ytrain]
        feed_dict = {self.x: xtrain, self.y: ytrain, self.sigma0: 1., self.initial_keep_prob: task['initial_keep_prob'],  self.is_training: False}
        z = tf.get_collection('log_network')[-1]
        batch_z = self.sess.run( z, feed_dict)
        data.append(batch_z)
    data = np.vstack(data)
    data = data.reshape(data.shape[0],-1)
    def normal_tc(c0):
        c1i = np.diag(1./np.diag(c0))
        p = np.matmul(c1i,c0)
        return - .5 * np.linalg.slogdet(p)[1] / c0.shape[0]
    c0 = np.cov( data, rowvar=False )
    tc = normal_tc(c0)
    print "Total correlation: %f" % tc


@ex.automain
def run():
    train()
    valid(load=False)




