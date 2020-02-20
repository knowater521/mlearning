"""
1.随机取k个中心点
2.计算所有点到中心点的距离
  将所有点分别放入中心点所在的簇
        更新中心点
            当中心点不变结束迭代
  迭代

"""
import numpy as np

def loadDataSet(filename):
    return np.loadtxt(filename, delimiter=",")

#取出k个中心点
def initCenter(dataset, k):
    #返回k个中心点
    centerIndex = np.random.choice(len(dataset), k, replace=False)
    return dataset[centerIndex]

#计算距离公式
def distance(x, y):
    return np.sqrt(np.sum((x-y)**2))

#主算法
def kmeans(dataset, k):
    #返回k个簇
    #初始化中心点
    centers = initCenter(dataset, k)
    n, m = dataset.shape
    #用于存储每个样本属于哪个簇
    clusters = np.full(n, np.nan)
    #迭代
    flag = True
    while flag:
        flag = False
        #计算所有点到簇中心点的距离
        for i in range(n):
            minDist, clustersIndex = 999999999.9, 0
            for j in range(len(centers)):
                dist = distance(dataset[i], centers[j])
                if dist<minDist:
                    #为样本分簇
                    minDist = dist
                    clustersIndex = j
            if clusters[i] != clustersIndex:
                clusters[i] = clustersIndex
                flag = True
        print(centers)
        #更新簇中心
        for i in range(k):
            subdataset = dataset[np.where(clusters==i)]
            centers[i] = np.mean(subdataset, axis=0)
    return clusters





if __name__ == "__main__":
    dataset = loadDataSet("data.txt")
    print(kmeans(dataset, 2))