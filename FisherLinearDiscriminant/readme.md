# Fisher Linear Discriminant

We will build up the Fisher Linear Discriminant step by step and generalize it for multiple classes
We can take any D-dimensional input vector and project it down to D′-dimensions
D : original input dimensions 
D′: projected space dimensions
D′ < D

To project the data into D′ = 1 dimension, we can pick a threshold t to separate the classes in the new space
Given an input vector x:
- x belongs to class C1 (class 1), if predicted y >= t, where y = wT x
- otherwise, it is classified as C2 (class 2)

## K = 2 Classes

D = 2 and we want to reduce the original data dimensions from D = 2 to D′ = 1

- Compute the mean vectors of the two classes and consider using the class means as a measure of separation
- Project the data onto the vector w joining the 2 class means

During such projection, there can be information loss
For FLD, it looks for a large variance among the dataset classes and a small variance within each of the dataset classes

w ∝ Sw^−1 * ( m2 − m1 )

## K > 2 Classes

Setup:
- Mean vector, mk
- Within class variance, Sk
- Total within class variance, Sw
- Between class variance, Sb

Then we can have the project vector
w = max D′ ( eig ( Sw^−1 * Sb ))

Next, we create a discriminant, we first can use a multivariate Gaussian distribution over a D-dimensional input vector x for each class K

The class priors P(Ck) probabilities are also calculated

Once we have the Gaussian parameters and also the priors for each class, then we estimate the class-conditional densities P(x|Ck,μk,Σk) for each class k

We then assign the input data x to the class k ∈ K with the largest posterior

# MNIST dataset

D = 784 dimensions
Do a projection to D′ = 2 or D′ = 3

- Implement the FLD for high-dimension to low-dimension projection with multivariant Gaussian
- Split the MNIST dataset into training and testing
- Report the accuracy on test set with D = 2 and D′ = 3
- Make plots for testing data with a 2D and 3D plots
