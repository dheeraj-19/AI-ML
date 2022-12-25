# dmancha-a4
# B 551 Assignment 4: Machine Learning


# Part 1: K-Nearest Neighbors Classification

## Formulation of the problem:

The task given here is to implement K-Nearest Neighbors classification

In simple terms, this classification algorithm tries to predict the correct class for a test sample, based on its K closest neighbors.

This can be done in two ways:
1. A simple majority vote of the K nearest neighbors
2. Add weights to the neighbors, so that the closest one's contribute more to the prediction

The distance between the data points can be calculated in two ways:
1. Manhattan distance
2. Euclidean distance

## Implementation:

1. Distance Functions:

		Started by implementing the two distance functions in _utils.py_ : _euclidean_distance_ and _manhattan_distance_
		Calculated the distance between two vestors based on the math formula's

2. _fit_ Function:

		Assigned the training and test data points to the class variables

3. _predict_ Function:

		This will loop through all the data points in the test dataset.
		1. Call _get_k_n_ function:
			Input: Test data point
			Output: Returns the distances of the test data point to all the points in the train data set
		2. Based on these distances find the K nearest points and their respective classes
		3. Based on the weight attribute call the respective function:
			1. Uniform weight: _weight_uniform_
				Implements a simple majority vote of the K nearest neighbors
				Selects the correct class for the test data point
			2. Distance weight: _weight_distance_
				Calls the function: _find_weights_
					Gets the weight for each neighbour
					These weights are proportional to the inverse of the distance from the test data point to the neighbor
				After getting the weights it chosses the class which has the maximum weight
		4. Returns the predicted classes for all the test data points



# Part 2: Multilayer Perceptron Classification

## Formulation

The task given here is to implement a feedforward fully-connected multilayer perceptron classifier with one hidden layer

A multilayer perceptron consists of neurons that are arranged in multiple layers that connect to each other and create a network called as neural network.

A neuron is a simple computational unit that takes an input signal and produces and output signal using an activation function.

The inputs to neurons are weighted and a bias is also added to them.

These weighted inputs are then sent to the activation function.

The different activation functions that we have to use in the problem are:
1. Identity
2. Sigmoid
3. Tanh
4. Relu

The network we have to implement here will have exactly three layers:
1. Input Layer: It has the same number of nodes as the number of features that each data point in the train dataset has
2. Hidden Layer: The number of neurons are dynamic 
3. Output Layer: This outputs a probability indicating the chance that a data point can be a specific target class

We also have to encode the training data using the one-hot-encoding mechanism

## Implementation:

The implementation of the multilayer perceptron follows the below steps:
1. Prepare the data
2. Training the data using batch gradient descent
3. For each iteration, first perform forward propagation
4. Calculate the cross entropy error
5. Perform backward propagation, take the error back to the input layer
6. Adjust the weights accordingly
7. Predict the classes of the test data by performing forward propagation

Code Changes
1. Activation Functions:

		Started by implementing the activation functions in _utils.py_:
		1. _identity_
		2. _sigmoid_
		3. _tanh_
		4. _relu_

2. Implmented _one_hot_encoding_ functions in _utils.py_
3. __initialize_ Function:
		
		1. Called the _one_hot_encoding_ function to encode the class labels for the training data
		2. Assign this to the class variable
		3. Initialize weights and bias parameters:
			Between Input layer and Hidden Layer
			Between Hidden layer and Output Layer
			
4. _fit_ Function:

		1. Calls the Initialize function
		2. Runs a loop for the number of iterations to be executed by the network
		3. Forward Propagate:
			Calculate Input to hidden layer, by using the weights and bias, and train data points
			Calculate Outup from hidden layer, by calling the activation function: [identity, sigmoid, tanh, relu]
			Calculate Input to output layer, by using the weights and bias
			Calculate Onutput from output layer, by calling the activation function: softmax
		4. Calculate the error loss by calling the cross_entropy function, that was implemented in utils.py
		5. Backward Propagate:
			Calculate the gradient descent of the weights and bias based on the loss and go back to the input layer
		6. Update the weights and bias based on the gradient descent's 

5. _predict_ Function:

		1. Forward Propagate:
			Calculate Input to hidden layer, by using the weights and bias, and test data points
			Calculate Outup from hidden layer, by calling the activation function: [identity, sigmoid, tanh, relu]
			Calculate Input to output layer, by using the weights and bias
			Calculate Onutput from output layer, by calling the activation function: softmax
			
		2. Return an argmax on the ouput from the output layer of the network, which will be the predicted class of the test data points



		
