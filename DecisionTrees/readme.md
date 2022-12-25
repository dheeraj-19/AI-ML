# Decision Trees

## Univariate

- Predict if a person has a college degree based on age and salary
- Build a decision tree for classifying whether a person has a college degree by greedily choosing threshold splits that maximize information gain

## Multivariate

- A multivariate decision tree is a generalization of univariate decision trees
- Where more than one attribute can be used in the decision rule for each split
- That is, splits need not be orthogonal to a feature’s axis
- For the same data, learn a multivariate decision tree where each decision rule is a linear classifier that makes decisions based on the sign of αxage + βxincome − 1

Trying to implement perceptron aprroach using :
- https://towardsdatascience.com/perceptron-algorithm-in-python-f3ac89d2e537
- https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/

Got the values of alpha and beta, but didn't quite understand how to implement it.
Was unable to find enough resources for its implementation.
