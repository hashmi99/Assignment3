
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
a = dataset.iloc[:, 0:1].values
b = dataset.iloc[:, 1:2].values
c = dataset.iloc[:, 2:3].values
X=a+b+c
y = dataset.iloc[:, 4].values



# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Expense Vs Profit (Linear Regression)')
plt.xlabel('Research Expense')
plt.ylabel('Profit')
plt.show()


# Predicting a new result with Linear Regression
s=lin_reg.predict([[900000]])
print(s)


