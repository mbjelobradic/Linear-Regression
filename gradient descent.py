import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math, copy

#Created Dataframe
df=pd.read_excel(r"path of excel document", usecols=["column name in excel file for label x"]).values
df1=pd.read_excel(r"path of excel document", usecols=["column name in excel file for label y"]).values

#x_train is the input variable and y_train is the target
x_train=np.array (df[:10]) #first 10 data points
y_train=np.array (df1[:10]) #first 10 data points

print(f"x_train=\n{x_train}")
print(f"\ny_train=\n{y_train}")

#m is the number of training examples
print (f"\nx_train.shape: {x_train.shape}")
m= x_train.shape[0]
print (f"Number of training example is: {m}\n")




def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    
    Parameters
    ----------
    x : Data (m examples).
    y : target values.
    w, b : Model parameters.

    Returns
    -------
    total_cost : The cost of using w, b as the parameters for linear regression to fit the data points in x and y.

    """
    
    #number of training examples
    m=x.shape[0]
    
    cost_sum=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=(f_wb-y[i])**2
        cost_sum=cost_sum+cost
    total_cost=(1/(2*m))*cost_sum
    
    return total_cost

def compute_gradient (x, y, w, b):
    
    """
    
    x : Data (m examples).
    y : target values.
    w, b : Model parameters.

    Returns
    -------
    dj_dw : The gradient of the cost with respect to the parameter w.
    dj_db : The gradient of the cost eith respect to the parameter b.

    """
    
    #Number of training examples
    m=x.shape[0]
    dj_dw=0
    dj_db=0
    
    for i in range (m):
        f_wb=w*x[i] + b
        dj_dw_i = (f_wb-y[i])*x [i]
        dj_db_i = f_wb-y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    
    return dj_dw, dj_db


def gradient_descent (x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w and b by taking num_iters gradient steps with learning rate alpha.

    Parameters
    ----------
    x : Data.
    y : Target values.
    w_in : Initial value of model parameters w.
    b_in : Initial value of model parameter b.
    alpha : Learning rate.
    num_iters : Number of iterations to run gradient descent.
    cost_function : Function to call to produce cost.
    gradient_function : Function to call to produce gradient.

    Returns
    -------
    w : Updated value of parameter after running gradient descent.
    b : Updated value of parameter after running gradient descent.
    J_history : History of cost values.
    p_history : History of parameters w and b.

    """
    
    #An array to store cost J and w at each iteration
    J_history =[]
    p_history=[]
    b=b_in
    w=w_in
    
    for i in range (num_iters):
        #Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        #Update Parameters using equation
        b=b-alpha*dj_db
        w=w-alpha*dj_dw
        
        #Save cost J at each iteration
        if i<100000:   #prevent resource exhaustion
            J_history.append (cost_function(x,y,w,b))
            p_history.append ([w,b])
            
        #Print cost every at intervals 10 times or as many iterations if <10
        if i% math.ceil(num_iters/10)==0:
            print(f"Iteration {i}, Cost {J_history[-1]}",
                  f"w: {w}, b: {b}")
    return w, b, J_history, p_history #Return w, b and J,w history

#Initialize parameters, it could be any numbers
w_init=0
b_init=0

#Some gradient descent settings
iteration=10000
tmp_alpha=1.0e-2

#Run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iteration, compute_cost, compute_gradient)
print (f"\n(w,b) found by gradient descent: ({w_final.round (2)}, {b_final.round(2)})") #final values for parameters w and b are expressed on 2 decimal points


def compute_model_output (x, w, b):
    """
    Computes the predicition of linear model.
    Parameters
    ----------
    x : Data.
    w,b : Model parameters.


    Returns
    -------
    f_wb : Model predicition.

    """
    
    m=x.shape[0]
    f_wb = np.zeros (m)
    for i in range(m):
        f_wb[i]=w*x[i]+b
        
    return f_wb

tmp_f_wb = compute_model_output(x_train, w_final, b_final,)

#Plot our model predicition
plt.plot (x_train, tmp_f_wb, c='b', label="Predicted values")

#Plot the data points
plt.scatter (x_train, y_train, marker="x", c="r", label="Actual values")
plt.ylabel ('y label')
plt.xlabel ('x label')
plt.legend()
plt.show()