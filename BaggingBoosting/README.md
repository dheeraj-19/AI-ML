# Bagging and Boosting

We implement the boosting algorithm AdaBoost as well as a Bagging
In both cases using decision trees as the base classifiers

## AdaBoost
### Algorithm
    •1: Initiallize a vector of 𝑛 uniform instance weights 𝑤!
    •2: for 𝑡 = 1, ... , 𝑇
    •3: Train model ℎ" on 𝑋, 𝑦 with weights 
    •4: Compute the weighted training error of ℎ
    •5: Choose 𝛽
    •6: Update all instance weights:
    •7: Normalize 𝑤 to be a distribution
    •8: end for
    •9: Return the hypothesis

- Write a function to implement the AdaBoost algorithm
- The input to the function is the training and test sets, as well as the number of rounds of boosting T
- Run AdaBoost for T rounds, using the decision-tree algorithm as the base learner
- The function should return the predictions of the final combined classifier on the given training and test examples, as well as the training and test error rate of the combined classifier following each of the T rounds

## Bagging
- A different method for combining decision trees or other base classifiers
- Similar to boosting, the base learning algorithm is run repeatedly in a series of rounds, but the manner in which the base learner is called is different
- On each round, the base learner is trained on what is often called a ”bootstrap replicate” of the original training set
- This ”bootstrap replicate” is formed by repeatedly selecting uniformly at random and with replacement m examples from the original training set
- A base classifier is then trained on this replicate, and the process continues
- After T rounds, a final combined classifier is formed which simply predicts with the majority vote of all of the base classifiers

## Test on three real world datasets


### Letter dataset:

    The dataset is available at https://archive.ics.uci.edu/ml/datasets/letter+recognition
    
    - Contains descriptions of the characters ”C” and ”G” 
    - The goal is to distinguish between these two letters
    - The class label is either ”C” or ”G”
    - There are 16 attributes for things like the width of the letter and the total number of pixels turned on
    - There are 500 training and 1009 test examples
    
    
### Credit dataset

    The dataset is available at https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    
    - Classifies people described by a set of attributes as good or bad credit risks
    - There are 20 attributes encoding credit history, purpose of the loan, employment status, etc
    - There are 400 training and 600 test examples
    
    
### Spam dataset

    The dataset is available at https://archive.ics.uci.edu/ml/datasets/spambase
    
    - Classifies email messages as spam or ham
    - The 57 attributes mainly encode the number of times that certain words or characters occur
    - There are 1000 training and 3601 test examples



    



