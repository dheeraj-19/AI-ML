# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: Dheeraj Deepak Manchandia -- dmancha
#
# Based on skeleton code by CSCI-B 551 Fall 2022 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def get_k_n(self, x_test_r):
        
        distances = []
        
        for x_train_r in self._X:
            d = self._distance(x_test_r, x_train_r)
            distances.append(d)
        
        return distances
        
    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        
        self._X = X
        self._y = y
        
        return

    def find_weights(self, knn_class):
    
        s = 0
        weights = [[y,0] for x,y in knn_class]
        
        for x in range(len(knn_class)):
            if knn_class[x][0] == 0:
                weights[x][1] = weights[x][1] + float('inf')
            else:
                weights[x][1] = weights[x][1] + (1.0 / knn_class[x][0])
		
        return weights
		
    def weight_distance(self, knn_class):
        
        weighted_classes = self.find_weights(knn_class)
        
        y_pred = {}
        
        for x in weighted_classes:
            if x[0] in y_pred:
                y_pred[x[0]] = y_pred[x[0]] + x[1]
            else:
                y_pred[x[0]] = x[1]
        
        return max(y_pred, key = y_pred.get)    
    
    def weight_uniform(self, knn_class):
    
        y_pred = [y for x,y in knn_class]
        
        return max(set(y_pred), key=y_pred.count)

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        
        
        y_pred_class = []
        for x_test_r in X:
            knn = self.get_k_n(x_test_r)
            knn_class = [(x,y) for x,y in sorted(zip(knn,self._y))]
            y_pred = knn_class[:self.n_neighbors]
            
            if self.weights == 'uniform':
                y_pred_class.append(self.weight_uniform(y_pred))
            else:
                y_pred_class.append(self.weight_distance(y_pred))
    
        return np.array(y_pred_class)
