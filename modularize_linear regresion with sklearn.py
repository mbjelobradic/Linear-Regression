import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def load_data(file_path, features, target, n_samples=9):
    data = pd.read_excel(file_path, usecols=features + [target])
    x_train = data[features].iloc[:n_samples].values
    y_train = data[target].iloc[:n_samples].values
    return x_train, y_train

def train_and_evaluate_model(x_train, y_train):
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x_train)

    sgdr = SGDRegressor(max_iter=100000)
    sgdr.fit(x_norm, y_train)

    y_pred_sgd = sgdr.predict(x_norm)
    r2 = r2_score(y_train, y_pred_sgd)
    
    return sgdr, y_pred_sgd, r2

def visualize_results(x_train, y_train, y_pred):
    plt.scatter(x_train, y_train, color='blue', label='Actual Values')
    plt.scatter(x_train, y_pred, color='red', label='Predicted Values')
    plt.plot(x_train, y_pred, color='red', label='Regression Line')
    plt.xlabel('feature column')
    plt.ylabel('target column')
    plt.legend()
    plt.title('Actual vs. Predicted Values')
    plt.show()

if __name__ == "__main__":
    file_path = r"file_path"
    features = ["fetaure column"]
    target = "target column"

    x_train, y_train = load_data(file_path, features, target)
    model, y_pred, r2 = train_and_evaluate_model(x_train, y_train)

    print(f"R-squared score: {r2:.2f}")
    visualize_results(x_train, y_train, y_pred)

