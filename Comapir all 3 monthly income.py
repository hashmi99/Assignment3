

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('monthlyexp vs incom.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values



# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.suptitle('Expense Vs Income')
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
a=lin_reg.predict([[50]])
print(a)

# Predicting a new result with Polynomial Regression
b=lin_reg_2.predict(poly_reg.fit_transform([[50]]))
print(b)
# Predicting a new result
c=y_pred = regressor.predict([[50]])
print(c)
