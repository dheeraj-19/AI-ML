# Naive Bayes

People’s name can be connected to which country he/she comes from
Here we have 4000 (fake) names: Japanese, American, Arabic, and Greek

## Goal:
Implement a NB classifier that can make a prediction given a new name

## Implementation:

- Merge all the names
- Split them into training (70%) and testing (30%) with shuffle = True
- Use CountVectorizer from sklearn.feature extraction.text to vectorize input names as a preprocessing step
- With vectorized representation of the input, then we can implement the algorithm
- Report testing accuracy with the algorithm
