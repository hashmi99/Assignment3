

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('global_co2.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.suptitle('Global Co2')
ax1.set_title('Linear Regression')
ax1.scatter(X, y, color = 'red')
ax1.plot(X, lin_reg.predict(X), color = 'blue')
ax2.set_title('Ploynomial Regression')
ax2.scatter(X, y, color = 'red')
ax2.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
ax3.set_title('Decision Tree')
ax3.scatter(X, y, color = 'red')
ax3.plot(X, regressor.predict(X), color = 'blue')
ax4.scatter(X, y, color = 'red')
ax4.set_title('Data Set')

for ax in fig.get_axes():
    ax.label_outer()


# Predicting a new result with Linear Regression
m=lin_reg.predict([[15000]])
print(m)

# Predicting a new result with Polynomial Regression
n=lin_reg_2.predict(poly_reg.fit_transform([[15000]]))
print(n)

# Predicting a new result
o=y_pred = regressor.predict([[15000]])
print(o)

