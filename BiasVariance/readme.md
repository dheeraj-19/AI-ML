# Bias Variance Decomposition

The squared error can be decomposed into bias, variance and noise

We will create a data set for which we can approximately compute this decomposition
The function '_toydata_' generates a binary data set with class 1 and 2. Both are sampled from Gaussian distributions

We will implement four functions: 

### computeybar: Noise
    - Compute y bar(x)
    - Compute the probability
    - Then use Bayes rule
    
### computehbar: Bias
    - Compute h bar
    - We cannot compute the expected value, but we can approximate it by training many hD and averaging their predictions
    - Edit the function computehbar: average over n models different hD, each trained on a different data set of n inputs drawn from the same distribution
    - Also call toydata to obtain more data sets
    
### computevariance: Variance
    - We need to compute the expected value
    - Once again, we can approximate this term by averaging over n models models
    
### biasvariancedemo: Demo
    - Implement a plotting function that shows how the error decomposes into bias, variance and noise when regularization constant λ increases

