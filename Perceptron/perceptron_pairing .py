import numpy as np 

class PercepTron(object):
    def __init__(self):
        self.b = 0
    
    def fit(self, X, y):
        a = 1 #learning rate
        size = len(X)
        alp = np.zeros(size)
        
        gram = np.dot(X, X.T) #Gram矩阵
        print("gram:",gram)

        b = self.b

        reset = True
        while(reset):
            reset = False
            for i in range(size):
                y_hat = y[i] * (np.sum(alp * y * gram[i]) + b)
                print("y_hat: ",y_hat)
                if(y_hat<=0):
                    alp[i] = alp[i] + 1
                    b = b + y[i]
                    reset = True
        w = np.sum(alp * y * X.T)

        return w, b

if __name__ == "__main__":
    X  = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])

    model = PercepTron()
    w, b = model.fit(X, y)
    print("w = ", w, " b = ", b)