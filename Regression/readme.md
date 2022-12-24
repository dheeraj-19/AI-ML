# Sydney dataset

https://www.kaggle.com/shree1992/housedata

### The goal is to predict the ’price’ column. 

- For this task, we can ignore the date
- Determine which features are continuous vs. categorical. Drop rows without a valid sales price
-  Visualize the univariate distribution of each continuous variable, and the distribution of the target (any variables that that might require special treatment?)
- Visualize the dependency of the target on each continuous feature (2d scatter plot)
- Split the data in training and testing set
- Use ColumnTransformer to encode categorical variables
- Apply Regressions
Evaluate Linear Regression (OLS)
Ridge
Lasso
Elasticnet
