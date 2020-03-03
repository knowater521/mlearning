import numpy as np 
from sklearn.linear_model import Perceptron


if __name__ == "__main__":
    X  = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])

    model = Perceptron()
    model.fit(X, y)
    w = model.coef_
    b = model.intercept_
    
    print("w = ", w, " b = ", b)