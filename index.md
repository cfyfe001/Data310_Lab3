## **Question One:**

**An "ordinary least squares" (or OLS) model seeks to minimize the differences between your true and estimated dependent variable.  True/False**

True - according to our notes, we can obtain the slope and intercept of the best fit line by applying the OLS. We determine the values of the slope and intercept such that the sum of the square distances between the points and the line is minimal. Likewise, minimizing the square of these differences, also results in minimizing the 'normal' difference between the two variables.


## Question Two:

**Do you agree or disagree with the following statement:
In a linear regression model, all features must correlate with the noise in order to obtain a good fit. Agree/Disagree**

Disagree - noise is an independent error term and identically distributed with a zero mean. Linear regression models can still have a good fit without adjusting to all 'noise' points. 


## **Question Three:**

**Write your own code to import L3Data.csv into python as a data frame. Then save the feature values 'days online', 'views', 'contributions', 'answers' into a matrix x and consider 'Grade' values as the dependent variable. If you separate the data into Train & Test with test_size=0.25 and random_state = 1234. If we use the features of x to build a multiple linear regression model for predicting y then the root mean square error on the test data is close to:**
```markdown
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

df = pd.read_csv('drive/MyDrive/Colab Notebooks/L3Data.csv')
y = df['Grade'].values
x1 = df.loc[:, df.columns != 'Grade']
x = x1.drop(columns=['questions']).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1234)
lm = linear_model.LinearRegression()

model = lm.fit(x_train,y_train)
y_pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
rmse
```
Based on the code above, the root mean squared error is about 8.32448.


## Question Four:

**In practice we determine the weights for linear regression with the "X_test" data. True/False**

False - you must use the X_train data to form the weights/coefficients for your linear regression model. 
i.e. if model = lm.fit(x_train,y_train); then model.coef_ deals with training data


## Question Five:

**Polynomial regression is best suited for functional relationships that are non-linear in weights. True/False**

True - the main idea of polynomial regression is that there is a combination of different powers of the feature values; i.e. the weights are non-linear. 

After submission, I found this answer to be wrong - the correct answer is **False**. For functional relationships that have non-linear weights, non-linear regression models should be used.


## **Question Six:**

**Linear regression, multiple linear regression, and polynomial regression can be all fit using LinearRegression() from the sklearn.linear_model module in Python. True/False**

True - all of these regressions are able to use LinearRegression() to fit the data. In fact, a polynomial function is a linear combination of the powers of x.


## Question Seven:

**Write your own code to import L3Data.csv into python as a data frame. Then save the feature values 'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable. If you separate the data into Train & Test with test_size=0.25 and random_state = 1234, then the number of observations we have in the Train data is**

```markdown
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('drive/MyDrive/Colab Notebooks/L3Data.csv')
y = df['Grade'].values
x1 = df.loc[:, df.columns != 'Grade']
x = x1.drop(columns=['questions']).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1234)

x_train.shape
```
This result gives us (23, 4) so it can be concluded that there are 23 observations.


## Question Eight:

**The gradient descent method does not need any hyperparameters. True/False**

False - the gradient descent method requires "data," "starting_b," "starting_m," "learning_rate," and "num_iterations." Both learning_rate and num_iterations are hyperparametes, so this statement is false.


## **Question Nine:**

**To create and display a figure using matplotlib.pyplot that has visual elements (scatterplot, labeling of the axes, display of grid), in what order would the below code need to be executed?**

```markdown
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter(x_test[:,0], y_test, color="black", label="Truth")
ax.scatter(x_test[:,0], lm.predict(x_test), color="green", label="Linear")
ax.set_xlabel("Discussion Contributions")
ax.set_ylabel("Grade")

ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
ax.minorticks_on()
```
Based on this order, the graph will be correctly printed. The 'import' statement always comes first because it brings the required module into the notebook. Next, fig and ax are required to name other statements from following lines, so it comes second. Thirdly, the data points need to be formed prior to edits to the grid, so this group comes third. This last chunk puts the final touches on the graph, so it comes at the end.


## Question Ten:

**Which of the following forms is not linear in the weights?**

The first choice contains weights non linear in nature. For weights to be non linear, they must be raised to any degree higher than the first; this first choice is the only answer choice with this characteristic.
