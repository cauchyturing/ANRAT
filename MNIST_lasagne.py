# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:05:54 2016

@author: Stephen-Lu
"""
from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

from keras.datasets import mnist
import lasagne


def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
#    network = lasagne.layers.GlobalPoolLayer(network)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# ############################# Batch iterator ###############################
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def ANRAE(prediction, target_var, lmbda, rl):
    return T.mean(1.0/lmbda**2) * T.log(T.mean(T.exp(T.mean(lmbda**2) * 
    (1 - prediction[T.arange(target_var.shape[0]), target_var])**2))) + rl * T.abs_(1.0 / T.mean(lmbda))
# ############################## Main program ################################

model='cnn' 
num_epochs=432
np.random.seed(306)
# Load the dataset
print("Loading data...")
(X_train, y_train), (X_test, y_test) = mnist.load_data() 

X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.
X_val = X_train[50000:].reshape(-1, 1, 28, 28)
y_val = y_train[50000:]
X_train = X_train[:50000].reshape(-1, 1, 28, 28)
y_train = y_train[:50000]
X_test = X_test.reshape(-1, 1, 28, 28)
# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

# Create neural network model (depending on first command line parameter)
print("Building model and compiling functions...")
lmbda = 8.
lmbda = theano.shared(np.asarray([lmbda], dtype = theano.config.floatX), 'lmbda', borrow = True)

network = build_cnn(input_var)

# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the NRAE loss):
prediction = lasagne.layers.get_output(network)
#loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
#loss = loss.mean()
params = lasagne.layers.get_all_params(network, trainable=True)

loss = ANRAE(prediction, target_var, lmbda, rl = 0.01)
loss = loss.mean()

# Define the params /lambda
params.append(lmbda)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

# Create a loss expression for validation/testing. The crucial difference
# here is that we do a deterministic forward pass through the network,
# disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()
# As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

# Finally, launch the training loop.
print("Starting training...")
# We iterate over epochs:
val_max = -np.inf
epoch_max = 0
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, 128, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, 128, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))
    if val_acc / val_batches * 100 >= val_max:
        val_max = val_acc / val_batches * 100
        epoch_max = epoch
        
        

# After training, we compute and print the test error:
#%%
test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(X_test, y_test, 128, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))

# Optionally, you could now dump the network weights to a file like this:
# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
#
# And load them again later on like this:
# with np.load('model.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network, param_values)


#if __name__ == '__main__':
#    if ('--help' in sys.argv) or ('-h' in sys.argv):
#        print("Trains a neural network on MNIST using Lasagne.")
#        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
#        print()
#        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
#        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
#        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
#        print("       input dropout and DROP_HID hidden dropout,")
#        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
#        print("EPOCHS: number of training epochs to perform (default: 500)")
#    else:
#        kwargs = {}
#        if len(sys.argv) > 1:
#            kwargs['model'] = sys.argv[1]
#        if len(sys.argv) > 2:
#            kwargs['num_epochs'] = int(sys.argv[2])
#        main(**kwargs)