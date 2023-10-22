import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor #implements linear regression with stochastic gradient descent
from sklearn.preprocessing import StandardScaler #used for standardizing and normalizing the data
from sklearn.metrics import r2_score #implement r2
import matplotlib.pyplot as plt

#Load data
data=pd.read_excel(r"file_path", usecols=["feature_column","target_column"])
x_train=data ["feature_column"].iloc[:9].values #first 9 integers
y_train=data ["target_column"].iloc[:9].values

#Data normalization
scaler = StandardScaler()
x_norm = scaler.fit_transform (x_train.reshape(-1,1)) #we reshape x_train to ensure it's a 2D array required by StrdardScaler
print (f"Peak to Peak range by column in Raw X: {np.ptp(x_train)}")
print (f"Peak to Peak range by column in Normalized X: {np.ptp(x_norm)}")

#Create and fit the regression model
sgdr = SGDRegressor(max_iter=100000)
sgdr.fit(x_norm, y_train)
print (sgdr)
print (f" number of iterations completed: {sgdr.n_iter_}, number of weigh updates:  {sgdr.t_}")

#Model parameters
b_norm=sgdr.intercept_
w_norm = sgdr.coef_
print (f"model parameters: w: {w_norm}, b:{b_norm}")

#make prediction
y_pred_sgd = sgdr.predict (x_norm)
print(f"Predicted values:{y_pred_sgd[:4]}")
print(f"Actual values : {y_train[:4]}")

# Calculate the R-squared score
r2 = r2_score(y_train, y_pred_sgd)
print(f"R-squared score: {r2:.2f}")


# Visualize predicted values and actual values
plt.scatter(x_train, y_train, color='blue', label='Actual Values')
plt.scatter(x_train, y_pred_sgd, color='red', label='Predicted Values')
plt.plot(x_train, y_pred_sgd, color='red', label='Regression Line')
plt.xlabel('feature')
plt.ylabel('target')
plt.legend()
plt.title('Actual vs. Predicted Values')
plt.show()
