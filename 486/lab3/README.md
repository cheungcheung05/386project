# Lab 3 - Reguarlized Regression
  

### Directions
* When you accept the assignment in GitHub Classroom, a repository named `lab-03` will be automatically generated for you under the "stat486-winter2024" organization.
* Your personal repository for this homework can be found at `https://github.com/stat486-winter2024/lab-03-your_user_name`.
* Clone the Repository: 
    - Open your terminal and navigate to the directory where you want to download the repository.
    - Run `git clone [your repository URL]` to clone the repository onto your local machine.
* The cloning process will create a new directory named `lab-03-your_user_name`. Ensure that you perform all your work inside this directory.

* Add all your answers to the file `lab-03-answers.txt`.  
  * Replace the XXXX with your answer, but otherwise leave the file unchanged.  For free response answers, it's ok if you answers continues to the next line
* Once you've completed the assignment, be sure that your code and answers files are pushed to the GitHub repo. 
    
This assignment uses the ocean temperature data.  This data is a subset of the [oceanographic data](https://calcofi.org/data/oceanographic-data/bottle-database/) from [The California Cooperative Oceanic Fisheries Investigations](https://calcofi.org/)
You can find the data in [this](https://github.com/esnt/Data/tree/main/OceanicFisheries) repository (use the file called `ocean_data.csv`).
The goal is to predict the water temperature (T_degC).

For this assignment, use only the numerical features to predict 'T_deg_C' (that is, exclude 'Wea', 'Cloud_Typ', 'Cloud_Amt', and 'Visibility'). 

### 1. Prepare the data
* (a) Make a training and a test set using `train_test_split`.  Use `random_state=307` and the default test size (25% of the data).
* (b) Make a processing pipeline that
  * imputes missing values using `SimpleImputer(strategy='mean')`
  * creates polynomial features using `PolynomialFeatures(degree=2, include_bias=False)`
  * standardizes the features using `StandardScaler`
* (c) Process the training and test data.  Fit the pipeline using only the training data and the transform both the training and test data:
    ```
    Xtrain = pipe.fit_transform(Xtrain)
    Xtest = pipe.transform(Xtest)
    ```

### 2. Ridge Regression
* (a) Use `RidgeCV` to fit a ridge regression.  Use 10-fold cross-validation and search for possible alphas using `np.logspace(-6,6,30)`.
* (b) What is the optimal value for alpha?
* (c) What is the test MSE?
* (d) What is the value of the coefficient that is largest in magnitude (excluding the intercept)
* (e) How many coefficients are exactly equal to 0? 


### 3. Lasso Regression
* (a) Use `LassoCV` to fit a ridge regression.  Use 10-fold cross-validation and search for possible alphas using the default.
* (b) What is the optimal value for alpha?
* (c) What is the test MSE?
* (d) What is the value of the coefficient that is largest in magnitude (excluding the intercept)
* (e) How many coefficients are exactly equal to 0? 
 
 ### 4. Compare
* (a) For this problem, is Lasso or Ridge a better choice?  Justify your answer
* (b) Compare your best Lasso/Ridge model to the unregularized model.  Why or why not does the regularized model perform better?
