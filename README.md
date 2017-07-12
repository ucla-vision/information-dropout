# Information Dropout implementation

A TensorFlow implementation of **Information Dropout** [https://arxiv.org/abs/1611.01353].

Information Dropout is a form of stochastic regularization that adds noise to the activations of a layer in order to improve _disentanglement_ and _invariance_ to nuisances in the learned representation. The related paper _[Emergence of Invariance and Disentangling in Deep Representations](https://arxiv.org/abs/1706.01350)_ also establish strong theoretical and practical connections between the objective of Information Dropout (_i.e._, minimality, invariance, and disentanglement of the activations), the minimality, compression and generalization performance of the network weights, and the geometry of the loss function.

## Installation

This implementation makes use [TensorFlow](https://www.tensorflow.org) (tested with v1.0.1), and the Python package `sacred`.

To run the experiments, you will need a preprocessed copy of the CIFAR-10 and Cluttered MNIST datasets. You can download and uncompress them in the `datasets` directory using:

```
wget http://vision.ucla.edu/~alex/files/cifar10.tar.gz
wget http://vision.ucla.edu/~alex/files/cluttered.tar.gz
tar -xzf cifar10.tar.gz
tar -xzf cluttered.tar.gz
```

The CIFAR-10 dataset was preprocessed with ZCA using the included `process_cifar.py` script, while the Cluttered MNIST dataset was generated using [the official code](https://github.com/deepmind/mnist-cluttered) and converted to numpy format.

## Running the experiments

### CIFAR-10

To train a CNN on CIFAR-10 you can use commands in the following format:

```
./cifar.py train with dropout=information filter_percentage=0.25 beta=3.0
./cifar.py train with dropout=binary filter_percentage=0.25
```

The first command trains using information dropout, with parameter `beta=3.0`, and using a smaller network with only 25% of the filters. The second will train with binary dropout instead. All the trained models will be saved in the `models` directory using a unique name for the given configuration. To load and test a trained configuration, run

```
./cifar.py test with [...]
```

You can print the complete list of options using

```
./cifar.py print_config
```

Computing the **total correlation** of the layer is only supported for softplus activations using a log-normal prior at the moment. To train with softplus activations and compute the total correlation of the trained representation, run 

```
./cifar.py train with softplus filter_percentage=0.25 beta=1.0
./cifar.py correlation with softplus filter_percentage=0.25 beta=1.0
```

### Cluttered MNIST

Training on the Cluttered MNIST dataset uses a similar syntax:

```
./cluttered.py train with beta=0.5
```

To plot the **information heatmap** of each layer, which shows that Information Dropout learns to ignore nuisances and focus on information important for the task, use the following command. The results will be saved in the `plots` subdirectory.

```
./cluttered.py plot with beta=0.5
```

_Note: due to a slight change in the training algorithm, for this version of the code use `beta <= 0.5` to train._


## A minimal implementation

For illustration purpose, we include here a commented pseudo-implementation of a convolutional Information Dropout layer using ReLU activations.
```python
import tensorflow as tf
from tensorflow.contrib.layers import conv2d

def sample_lognormal(mean, sigma=None, sigma0=1.):
    """
    Samples from a log-normal distribution using the reparametrization
    trick so that we can backprogpagate the gradients through the sampling.
    By setting sigma0=0 we make the operation deterministic (useful at testing time)
    """
    e = tf.random_normal(tf.shape(mean), mean = 0., stddev = 1.)
    return tf.exp(mean + sigma * sigma0 * e)

def information_dropout(inputs, stride = 2, max_alpha = 0.7, sigma0 = 1.):
    """
    An example layer that performs convolutional pooling
    and information dropout at the same time.
    """
    num_ouputs = inputs.get_shape()[-1]
    # Creates a convolutional layer to compute the noiseless output
    network = conv2d(inputs,
        num_outputs=num_outputs,
        kernel_size=3,
        activation_fn=tf.nn.relu,
        stride=stride)
    # Computes the noise parameter alpha for the new layer based on the input
    with tf.variable_scope(None,'information_dropout'):
        alpha = max_alpha * conv2d(inputs,
            num_outputs=num_outputs,
            kernel_size=3,
            stride=stride,
            activation_fn=tf.sigmoid,
            scope='alpha')
        # Rescale alpha in the allowed range and add a small value for numerical stability
        alpha = 0.001 + max_alpha * alpha
        # Similarly to variational dropout we renormalize so that
        # the KL term is zero for alpha == max_alpha
        kl = - tf.log(alpha/(max_alpha + 0.001))
        tf.add_to_collection('kl_terms', kl)
    e = sample_lognormal(mean=tf.zeros_like(network), sigma=alpha, sigma0=sigma0)
    # Noisy output of Information Dropout
    return network * e

### BUILD THE NETWORK
# ...
# Computes the KL divergence term in the cost function
kl_terms = [ tf.reduce_sum(kl)/batch_size for kl in tf.get_collection('kl_terms') ]
# Normalizes by the number of training samples to make
# the parameter beta comparable to the beta in variational dropout
Lz = tf.add_n(kl_terms)/N_train
# Lx is the cross entropy loss of the network
Lx = cross_entropy_loss
# The final cost
cost = Lx + beta * Lz
```
