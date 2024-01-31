# Lab 2 - Pipelines and Gradient Descent

### Objective
The purpose of this lab is to get more practice with Scikit-learn, particularly with Scikit-learn pipelines, to review linear regression, and to gain a deeper understanding of gradient descent. 
  

### Directions
* When you accept the assignment in GitHub Classroom, a repository named `lab-02` will be automatically generated for you under the "stat486-winter2024" organization.
* Your personal repository for this homework can be found at `https://github.com/stat486-winter2024/lab-02-your_user_name`.
* Clone the Repository: 
    - Open your terminal and navigate to the directory where you want to download the repository.
    - Run `git clone [your repository URL]` to clone the repository onto your local machine.
* The cloning process will create a new directory named `lab-02-your_user_name`. Ensure that you perform all your work inside this directory.

* Add all your answers to the file `lab-02-answers.txt`.  
  * Replace the XXXX with your answer, but otherwise leave the file unchanged.  For free response answers, it's ok if you answers continues to the next line
* Once you've completed the assignment, be sure that your code, answers, and predictions files are pushed to the GitHub repo. 
    
## Section 1: Pipelines 
This part uses the ocean temperature data.  This data is a subset of the [oceanographic data](https://calcofi.org/data/oceanographic-data/bottle-database/) from [The California Cooperative Oceanic Fisheries Investigations](https://calcofi.org/)
You can find the data in [this](https://github.com/esnt/Data/tree/main/OceanicFisheries) repository.
The goal is to predict the water temperature (T_degC) 

### 1. Prepare the data
* (a) For parts, (1) - (5), use only the numerical features to predict 'T_deg_C' (that is, exclude 'Wea', 'Cloud_Typ', 'Cloud_Amt', and 'Visibility'). 
* (b) Make a training and a test set using `train_test_split`.  Use `random_state=307` and the default test size (25% of the data). 

### 2. Pipelines
The `Pipeline` object in scikit-learn is used to assemble several steps that can be cross-validated together while setting different parameters, streamlining the process of applying a sequence of transformations and a final estimator to data.  

The `Pipeline` object is instantiated with a list of tuples of the form `("model_nickname", Estimator(hyperparameters))`.  For example, here is a simple pipeline consisting of imputing missing values then fitting a linear regression model:
```{python}
pipe = Pipeline([
  ('impute','SimpleImputer('strategy'='median')),
  ('model', 'LinearRegression())
])
```
`Pipeline` objects can then be fitted and predicted from just like any other Scikit-Learn estimator:
```{python}
pipe.fit(Xtrain, ytrain)
yhat = pipe.predict(Xtest)
```
* (a) Use the `Pipeline` object from `sklearn.pipeline` to build a pipeline that does the following:
    * Imputes missing values (use `SimpleImputer` with `strategy="mean"`)
    * Adds 2rd order polynomial features of the Xs (use `PolynomialFeatures` with `degree=2` and `include_bias=False`) 
    * Standardizes the data (use `StandardScaler` with default options)
    * Fits a linear regression model (use `LinearRegression` with default options)
* (b) Fit your pipeline to the training data 
* (c) Report the training MSE
* (d) Report the test MSE
* (e) How does the test MSE compare to the variance of `ytest`?  What does this say about the predictability of your model?

### 3. Feature Importance
It is often desirable to look at the coefficients from a linear regression model.  If the features are all on the same scale, it is even possible to gauge feature importance by the magnitude of the coefficient.  Let's figure out how to extract this information from our pipeline.  

We can access attributes and methods from each step of the Pipeline by use the `.named_steps` attribute of the pipeline.  For example, using the same simple pipeline from above, 

```{python}
betas = pipe.named_steps['model'].coef_
``` 
will extract the coefficients of the linear regression into the variable `betas` and 
```{python}
features = pipe.named_steps['impute'].get_feature_names_out()
```
will return the names of the features.  For more complex pipelines, different steps of the pipeline might return different feature names, depending on the action of the step.  

* (a) Use the pipeline that you built in part (2) to order the features by the magnitude of their coefficients.
* (b) What are the features and coefficients with the three largest positive coefficients?
* (c) What are the features and coefficients with the three largest negative coefficients?
* (d) What is the estimated y-intercept of the fitted model?


### 4. Hyperparameter Tuning with Grid Search
One advantage of using a pipeline is that all the hyperparameters can be tuned and optimized at the same time using the function `GridSearchCV`.  In order to use this function, set up a pipeline specifying only the hyperparameters that need to specify but that you *don't* want to tune.  For example:
```{python}
pipe = Pipeline([
  ('impute','SimpleImputer()),
  ('poly', 'PolynomialFeatures(include_bias=False')),
  ('model', 'LinearRegression())
])
```
Next, set up the hyperparameters that you want to tweak and the values that you want to try in a dictionary.  The `key` will be the pipeline step short, two underscores, then the name of the hyperparameter you are tuning.  The `value` will be a list or tuple of options you want to try.  For example:
```{python}
params = {
  'impute__strategy':('mean','median'), 
  'poly__degree':(1,2,3)
}
``` 
Finally, instantiate and fit a GridSearchCV object specifying the pipeline object, parameter dictionary, scoring metric, number of CV folds:
```{python}
gs = GridSearchCV(pipe, param_grid=params, scoring='neg_mean_squared_error', cv=10)
gs.fit(X, y)
```
* (a) For this part, continue to only use the numeric variables but this time we'll fit a KNN model (fit the same pipeline, except substitute KNN for linear regression).
* (b) Use `GridSearchCV` with 10-fold cross validation and scoring metric negative MSE to find the optimal hyperparameters  from among the following options:
  - `strategy="mean"` vs `strategy="median"` with `SimpleImputer`
  - Polynomial terms of order 1, 2, and 3
  - Number of neighbors 5 to 100 by 5
  - KNN neighbor weights `uniform` vs `distance`
* (c) Print the best hyperparameter combinations and the best MSE
* (d) How long did it take you to fit the `GridSearchCV`? (*FYI, it took my computer about 30 seconds*)
* (e) Use the optimized pipeline to get predictions for the test data 
* (f) Report the test MSE 
* (g) Is this optimized pipeline any better at predicting water temperature than the pipeline in part (2)?


### 5. Randomized Grid Search
You might have noticed that `GridSearchCV` can be quite slow to run.  There is another function, `RandomSearchCV`, that works very similar but it often more desireable when there are many hyperparameters to tune and/or many possible options.  It won't always find the *most* optimal combination, but it is usually pretty close and runs much much faster.  You can read more about `RandomSearchCV` in [Chapter 2 of HOML](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/ch02.html#idm45720239663616).   
* (a) - (g) Repeat part (4) using `RandomSearchCV` instead of `GridSearchCV`
* (h) Compare the results of both methods.  Comment on your findings. 

### 6. Advanced Pipelines

* For this part, continue use all the numeric features, but also use "Wea", "Cloud_Typ", "Cloud_Amt" and "Visibility"
* Use the `Pipeline` function from `sklearn.pipeline` to build a pipeline that does the following:
    * For Numeric variables:
        * Imputes missing values using the mean with `SimpleImputer`
        * Adds 2rd order polynomial functions of the Xs using `PolynomialFeatures` (and `include_bias=False`) 
        * Standardizes the data using `StandardScaler`
    * For Categorical variable:
        * Imputes missing values with the mode using `SimpleImputer` (`strategy="most_frequent"`)
        * Creates dummy variables using `OneHotEncoder`.  Use arguments `sparse_output=False` and `handle_unknown='ignore'` 
        * Reduces dimensions by filtering dummy features with an F-test score using `SelectPercentile(f_regression, percentile=50)`
    * Uses `ColumnTransformer` to combine the numeric and categorical pipelines 
    * Fits a `KNeighborRegressor` model using `n_neighbors=20` and  `weights='distance'`
    * *Hint: refer to the [Scikit-Learn documentation](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py) for an example of how to construct and implement this pipeline*
* (a) Fit the pipeline to the training data
* (b) Report the training and test mse
* (c) Do the extra variables improve predictability? 

### 7. More Pipeline Functionality
There are more things that you can do with pipelines including hyperparameter tuning for Pipelines with `ColumnTransformer`'s,  making your own function transformers, and building custom classes.  You can learn more about these advanced features in HOML Chapter 2 in the section ["Prepare the Data for Machine Learning Algorithms"](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/ch02.html#idm45720245368288) and the [Scikit-Learn documentation](https://scikit-learn.org/stable/modules/compose.html#build-a-pipeline) with several examples.

* (a) Skim through the HOML section and some of the Scikit-Learn examples to get an idea of some of the capability of the pipeline.
* (b) What is something new that you learned about what pipelines can do? 

### 8. Make new Predictions
* (a) Of all the models that you've worked on in this assignment, choose the one that you think will make the best predictions on new data.  You can also tweak other hyperparameters or use a different model.  
* (b) Once you've selected your best model, predict the water temperature for the data in file called "ocean_data_newvalues.csv".  
  * Submit your predictions in a csv file called **"predictions.csv"**.  (I created a blank file with this name for you to add your predictions to.) 
  * This csv file should contain ONE column that has your predictions for the new data
  * This csv file should NOT contain a header (or column names).  The first row should be your prediction for the first value

---
---
## Section 2: Gradient Descent 
For this section, you will write your own gradient descent algorithm an apply it to an easy dataset.  

 
 * (a) Get the [ice cream data](https://github.com/esnt/Data/raw/main/CleanData/icecreamcone2.csv) from my GitHub data repo. The target is "logBV" and the features are the other 4 columns.  Process the data as follows: 
    * First, create the X and y objects:
    ```{python}
    X = df.drop(['logBV'], axis=1)
    y = df['logBV']
    ```
    * Next, standardize the features and add a column for the intercept
    ```{python}
    sc = StandardScaler()
    poly = PolynomialFeatures(degree=1)
    X = sc.fit_transform(X)
    X = poly.fit_transform(X)
    ```
    * Finally, in order to make the matrix algebra work, you'll also need to add a dimension of 1 to the target vector:
    ```{python}
    y = y.values.reshape(36,1)
    ```
    * The `@` sign is the symbol used for matrix multiplication and the `.T` attribute returns the matrix transpose.  For example, the following code will produce the normal equation estimates for the betas (assumes that numpy has been imported as `np`):
    ```{python}
    np.linalg.inv(X.T @ X) @ X.T @ y
    ```

  * (b) Write a function that uses  gradient descent to estimate the coefficients in a linear regression.  The function should take as input X (a 2D numpy array with the features), y (a 1D numpy array with the target), the learning rate, and the number of epochs (iterations) to run the algorithm.
    * **For your optimal learning, DON'T use ChatGPT, copilot, or similar tools.  Try to code it yourself.  That is the whole point of the exercise** 

  * (c) Apply your function to the ice cream data and report your estimate for the betas
  * (d) How do your GD results compare to the normal equation estimates?
 
 
