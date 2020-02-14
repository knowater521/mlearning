import numpy as np
import operator

def handle_data(dataset):
    x = dataset[:, :-1].astype(np.float)
    y = dataset[:, -1]
    return x, y
    

def knn_classifier(k, dataset, input):
    x, y = handle_data(dataset)
    #1.计算预测样本与数据集样本距离
    distance = np.sum((input - x)**2, axis=1)**0.5
    #2.所有距离从小到大排序
    sortedDist = np.argsort(distance)
    #3.计算前k个最小距离的类别个数
    countLabel = {}
    for i in range(k):
        label = y[sortedDist[i]]
        countLabel[label] = countLabel.get(label, 0) + 1
    #4.返回前k个最小距离中 个数最多的分类
    sortedLabel = sorted(countLabel.items(), key=operator.itemgetter(1), reverse=True)
    return sortedLabel[0][0]


if __name__ == "__main__":
    dataset = np.loadtxt("dataset.txt", dtype=np.str, delimiter=",")
    print(dataset)
    test = [2, 2]
    print(knn_classifier(3, dataset, test))