# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: Dheeraj Deepak Manchandia -- dmancha
#
# Based on skeleton code by CSCI-B 551 Fall 2022 Course Staff

import numpy as np
from math import sqrt

def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
	
    d = 0
	
    for x in range(len(x1)-1):
        d = d + (x1[x] - x2[x])**2
	
    return sqrt(d)


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """

    d = sum(abs(x-y) for x, y in zip(x1,x2))

    return d


def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    
    if not derivative:
        return x
    else:
        return np.ones(x.shape)


def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    
    if not derivative:
        return 1 / ( 1 + np.exp(-x))
    else:
        return sigmoid(x) * (1 - sigmoid(x))


def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    if not derivative:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    else:
        return 1 - (tanh(x) * tanh(x))


def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    if not derivative:
        return np.maximum(0,x)
    else:
        x[x<=0] = 0
        x[x>0] = 1
        return x


def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """
    
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return - (y / p) + (1 - y) / (1 - p)



def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """
    
    classes = np.unique(y)
    
    map_nos = {}
    for i in range(len(classes)):
        map_nos[classes[i]] = i
    
    one_hot_encode = []
    
    for i in y:
        t = list(np.zeros(len(classes), dtype = int))
        t[map_nos[i]] = 1
        one_hot_encode.append(t)
    
    return np.array(one_hot_encode)
