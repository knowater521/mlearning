import numpy as np 

class PercepTron(object):
    def __init__(self):
        self.b = 0
    
    def fit(self, X, y):
        a = 1 #learning rate
        size = len(X)
        w = np.array([0, 0])

        b = self.b

        reset = True
        while(reset):
            reset = False
            for i in range(size):
                y_hat = y[i] * (w.dot(X[i]) + b)
                if(y_hat<=0):
                    w = w + a * y[i] * X[i]
                    b = b + y[i]
                    reset = True
                print(w)
                print(b)


        return w, b

if __name__ == "__main__":
    X  = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])

    model = PercepTron()
    w, b = model.fit(X, y)
    print("w = ", w, " b = ", b)