import numpy as np

def createDataSet():
    dataSet = np.array([
        [0,0,0,0,'n'],
        [0,0,0,1,'n'],
        [1,0,0,0,'y'],
        [2,1,0,0,'y'],
        [2,2,1,0,'y'],
        [2,2,1,1,'n'],
        [1,2,1,1,'y']
    ])
    labels = np.array(['outlook', 'temperature', 'humidity', 'windy'])
    return dataSet, labels
def createTestSet():
    testSet = np.array([
        [0,1,0,0],
        [0,2,1,0],
        [2,1,1,0],
        [0,1,1,1],
        [1,1,0,1],
        [1,0,1,0],
        [2,1,0,1]
    ])
    return testSet

def dataset_entropy(dataset):#信息熵计算
    classLabel = dataset[:, -1]
    labelCount = {}
    for i in range(classLabel.size):
        label = classLabel[i]
        labelCount[label] = labelCount.get(label, 0) + 1
        """若不存在voteIlabel，则字典classCount中生成voteIlabel元素，并使其对应的数字为0，即
classCount = {voteIlabel：0}
此时classCount.get(voteIlabel,0)作用是检测并生成新元素，括号中的0只用作初始化，之后再无作用

当字典中有voteIlabel元素时，classCount.get(voteIlabel,0)作用是返回该元素对应的值，即0
        """
        ent = 0
    for k, v in labelCount.items():
        ent += - (v / classLabel.size * np.log2(v / classLabel.size))
    return ent

def splitDataSet(dataset, featureIndex):#切分数据
    subdataset = {} #划分后的子集
    featureValues = dataset[:, featureIndex]
    featureSet = set(featureValues)
    for i in range(len(featureSet)):
        newset = []
        for j in range(dataset.shape[0]):
            if featureSet[i] == featureValues[j]:
                newset.append(dataset[j, :])
        newset = np.delete(newset, featureIndex, axis=1)
        subdataset.append(np.array(newset))
    return subdataset
 
def splitDataSetByValue(dataset, featureIndex, value):#切分数据
    subdataset = []
    for example in dataset:
        if example[featureIndex] == value:
            subdataset.append(example)
    return np.delete(subdataset, featureIndex, axis=1)


def chooseFeature(dataset, labels):#选择最优特征
    featureNum = labels.size #特征个数
    #选择条件熵最小的
    minEntropy, bestFeatureIndex = 1, None
    n = dataset.shape[0] #样本总数
    for i in range(featureNum):
        featureEntropy = 0 #指定特征的条件熵
        #返回所有子集
        featureList = dataset[:, i]
        featureValues = set(featureList)
        for value in featureValues:
            subDataSet = splitDataSetByValue(dataset, i, value)
            featureEntropy += subDataSet.shape[0] / n * dataset_entropy(subDataSet)
        if minEntropy > featureEntropy:
            minEntropy = featureEntropy
            bestFeatureIndex = i
    return bestFeatureIndex

import operator
def mayorCount(classList):
    labelCount = {}
    for i in range(classList.size):
        label = classList[i]
        labelCount[label] = labelCount.get(label, 0) + 1
    sortedLabel = sorted(labelCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedLabel[0][0]


def createTree(dataset, labels):
    classList = dataset[:, -1]
    #当dataset中label列全相等，标记为相应特征并结束
    if len(set(dataset[:, -1]))==1:
        return dataset[:, -1][0]
    #当dataset某个属性上取值一样或label为空（无法继续切分）
    if label.size==0 or len(dataset[0])==1:
        return mayorCount(classList)
    #建树
    bestFeatureIndex = chooseFeature(dataset, labels)
    bestFeature = labels[bestFeatureIndex]
    dtree = {bestFeature:{}}
    featureList = dataset[:, bestFeatureIndex]
    featureValues = set(featureList)

    for value in featureValues:
        subdataset = splitDataSetByValue(dataset, bestFeatureIndex, value)
        sublabels = np.delete(labels, bestFeatureIndex)
        dtree[bestFeature][value] = createTree(subdataset, sublabels)
    return dtree

def predict(tree, labels, testData):
    rootNa = list(tree.keys())[0]
    rootValue = tree[rootNa]
    featureIndex = list(labels).index(rootNa)
    classLabel = None
    for key in rootValue.keys():
        if testData[featureIndex] == int(key):
            if type(rootValue[key]).__name__ == "dict":
                classLabel = predict(rootValue[key], labels, testData)
            else:
                classLabel = rootValue[key]
    return classLabel

def predictAll(tree, labels, testSet):
    classLabels = []
    for i in testSet:
        classLabels.append(predict(tree, labels, i))
    return classLabels

if __name__ == "__main__":
    print("start")
    dataset, label = createDataSet()
    #print(dataset_entropy(dataset))
    tree = createTree(dataset, label)
    #testData = [0, 2, 1, 0]
    #print(predict(tree, label, testData))
    testSet = createTestSet()
    print(predictAll(tree, label, testSet))
