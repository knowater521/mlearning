import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn_cla

if __name__ == "__main__":
    dataset = np.loadtxt("dataset.txt", dtype=np.str, delimiter=",")
    x = dataset[:, :-1].astype(np.float)
    y = dataset[:, -1]

    module = knn_cla()
    module.fit(x, y)

    test = [[2.6, 4.8]]
    result = module.predict(test)
    print(result)