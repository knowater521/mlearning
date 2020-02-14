import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
def weights(x, y):
    m, n = x.shape
    theta = np.random.rand(n)
    alpha = 0.001
    cnt = 0
    max_iter = 50000
    threshold = 0.01
    while cnt<max_iter:
        cnt += 1
        diff = np.full(n, 0)
        for i in range(m):
            diff = (y[i] - sigmoid(theta.T@x[i]))*x[i]
            theta = theta + alpha * diff
        if (abs(diff)<threshold).all():
            break
        if(cnt%100==0):
            print("decent %d" %(cnt))
    return theta

if __name__ == "__main__":

    x_train = np.array([
        [1, 2.697, 6.254],
        [1, 1.872, 2.014], 
        [1, 2.312, 0.812],
        [1, 1.983, 4.990],
        [1, 2.321, 0.812],
        [1, 2.215, 1.561],
        [1, 1.659, 2.932],
        [1, 0.865, 7.362],
        [1, 1.685, 4.763],
        [1, 1.786, 2.523]
    ])
    y_train = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1])

    print(weights(x_train, y_train))