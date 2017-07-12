from __future__ import print_function
import sacred
from sacred import Ingredient

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
import os

task_ingredient = Ingredient('task')

@task_ingredient.config
def cfg():
    batch_size = 100
    optimizer = 'adam'
    learning_rate = 1e-3
    learning_rate_drop = 0.1
    _load = False
    start_epoch = 0
    drop1 = 40
    drop2 = 80
    end_epoch = 120
    drop3 = end_epoch + 1
    _save = True
    _always_test = False
    _always_valid = False
    _shuffle = True
    _logging = True

    name = 'task'

    _model_dir = 'models'
    _log_dir = 'logs'
    _log_name = None

class Stats(object):
    def __init__(self, name=None):
        self.reset()
        self.history = []
        self.name = name

    def __getitem__(self, k):
        return self.stats[k]/self.stats_count[k]

    def reset(self):
        import collections
        self.stats = collections.OrderedDict()
        self.stats_count = collections.OrderedDict()

    def push(self, **kwargs):
        for k, v in kwargs.iteritems():
            self.stats[k] = self.stats.get(k, 0.) + v
            self.stats_count[k] = self.stats_count.get(k, 0) + 1

    def push_epoch(self):
        epoch_stats = {k:v/self.stats_count[k] for k, v in self.stats.iteritems()}
        self.history.append(epoch_stats)
        self.print_epoch()
        self.reset()
        return self.history

    def print_epoch(self):
        print('=== ', end='')
        if self.name is not None:
            print('%s: ' % self.name.capitalize() , end='')
        print( '%3d: ' % len(self.history), end='')
        for k in self.stats.iterkeys():
            print('%s: %.3f ' % (k, self[k]), end='')
        print('===') 

class Task(object):
    def __init__(self):
        pass
    
    @task_ingredient.capture
    def get_name(self,name, _run):
        from sacred.commands import _iterate_marked, ConfigEntry
        # from sacred import ConfigEntry
        def format_entry(entry):
            if not (entry.typechanged or entry.added or entry.modified):
                return ''
            if isinstance(entry, ConfigEntry) and entry.key[0] != '_':
                return '.' + entry.key + "=" + str(entry.value)
            else:  # isinstance(entry, PathEntry):
                return ''
        for path, entry in _iterate_marked(_run.config, _run.config_modifications):
            name = name + format_entry(entry)
        return name

    @task_ingredient.capture
    def get_model_file(self, _model_dir):
        return os.path.join(_model_dir, self.get_name()+'.ckpt')

    @task_ingredient.capture
    def get_log_file(self, _log_dir, _log_name):
        name = self.get_name() if _log_name is None else _log_name
        return os.path.join(_log_dir, name + '.p')
        
    @task_ingredient.capture
    def get_valid_batch(self, batch_size):
        return next(self.iterate_minibatches('valid'))

    def build_placeholders(self):
        pass

    def build_loss(self):
        self.loss = tf.constant(0.)

    @task_ingredient.capture
    def build_training_op(self, optimizer, _log):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], self.loss)
        else:
            loss = self.loss
        with tf.name_scope('optimizer'):
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            if optimizer=='adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif optimizer=='momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9, use_nesterov=True)
            else:
                raise ValueError("Invalid optimizer option")
            self.train_op = optimizer.minimize(loss, global_step=self.global_step)

    @task_ingredient.capture
    def create_dirs(self, _model_dir, _log_dir):
        if not os.path.exists(_model_dir):
            os.makedirs(_model_dir)
        if not os.path.exists(_log_dir):
            os.makedirs(_log_dir)

    @task_ingredient.capture
    def initialize(self, _load, _log_dir, _logging, _run, _seed):
        self.create_dirs()
        _run.info['log_file'] = self.get_log_file(_log_dir=_log_dir)
        _run.info['logging'] = _logging
        print(_run.info['log_file'])

        self.sess = tf.Session()
        self.sess.graph.as_default()
        tf.set_random_seed(_seed)
        with self.sess.as_default():
            self.global_step = tf.Variable(0, trainable=False)
            self.build_placeholders()
            self.build_loss()
            self.build_training_op()

            self.saver = tf.train.Saver(var_list=[v for v in tf.global_variables() if 'Momentum' not in v.name])
            if _load:
                self.load()
            else:
                tf.global_variables_initializer().run()

    def train_batch(self, batch, stats):
        pass

    def valid_batch(self, batch, stats):
        pass

    def end_epoch(self):
        pass

    @task_ingredient.capture
    def iterate_minibatches(self, dataset_name, batch_size, _shuffle):
        inputs = self.dataset[dataset_name]
        num_examples = inputs.shape[0] if not isinstance(inputs, (list,tuple)) else inputs[0].shape[0]
        a = np.random.permutation(num_examples) if _shuffle else np.arange(num_examples)
        for start_idx in range(0, num_examples - batch_size + 1, batch_size):
            # excerpt = [slice(start_idx, start_idx + batch_size)]
            excerpt = a[start_idx:start_idx+batch_size]
            if isinstance(inputs, (list,tuple)):
                yield [x[excerpt] for x in inputs]
            else:
                yield inputs[excerpt]

    @task_ingredient.capture
    def train(self, drop1, drop2, drop3, start_epoch, end_epoch, learning_rate, learning_rate_drop, batch_size, _save, _always_valid, _always_test, _log, _run):
        self.train_stats = Stats('train')
        self.valid_stats = Stats('valid')
        self.test_stats = Stats('test')
        try:
            for i in xrange(start_epoch, end_epoch):
                if i < drop1:
                    self.current_learning_rate = learning_rate
                elif i < drop2:
                    self.current_learning_rate = learning_rate * learning_rate_drop
                elif i < drop3:
                    self.current_learning_rate = learning_rate * learning_rate_drop**2
                else:
                    self.current_learning_rate = learning_rate * learning_rate_drop**3

                for batch in self.iterate_minibatches('train'):
                    self.train_batch(batch, self.train_stats)
                _run.info['train'] = self.train_stats.push_epoch()
                if _always_valid:
                    for batch in self.iterate_minibatches('valid'):
                        self.valid_batch(batch, self.valid_stats)
                    _run.info['valid'] = self.valid_stats.push_epoch()
                if _always_test:
                    for batch in self.iterate_minibatches('test'):
                        self.valid_batch(batch, self.test_stats)
                    _run.info['test'] = self.test_stats.push_epoch()
                self.end_epoch()
                if _save and i % 10 == 0:
                    self.save()
        except KeyboardInterrupt:
            _log.warn("Experiment interrupted by user.")
        finally:
            if _save:
                self.save()


    @task_ingredient.capture
    def valid(self, _run):
        self.valid_stats = Stats('valid')
        for batch in self.iterate_minibatches('valid'):
            self.valid_batch(batch, self.valid_stats)
        _run.info['valid'] = self.valid_stats.push_epoch()

    @task_ingredient.capture
    def load(self):
        model_file = self.get_model_file()
        self.saver.restore(self.sess, model_file)
        print('Loaded %s' % model_file)

    @task_ingredient.capture
    def save(self):
        model_file = self.get_model_file()
        print('Model saved in file: %s' % self.saver.save(self.sess, model_file))

