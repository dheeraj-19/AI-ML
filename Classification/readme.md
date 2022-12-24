# Classification

## ’credit-g’ dataset

Download the dataset with ’fetch openml(’credit-g’)’ and see its description at https://www.openml.org/d/31

- Determine which features are continuous and which ones are categorical
- Visualize the univariate distribution of each continuous variable, and the distribution of the target
- Split the data in training and testing set
- Preprocess the data (such as treatment of categorical variables)
- Evaluate initial Logistic Regression model (directly use the function)
- Use ColumnTransformer to encode categorical variables
- Evaluate classifications:
    1. Logistic Regression
    2. Linear Support Vector Machines
    3. Nearest neighbors
