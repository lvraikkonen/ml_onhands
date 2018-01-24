# Decision tree building using ID3

import numpy as np

class DecisionTree(object):
    def __init__(self, mode="ID3"):
        self._tree = None
    
    def _calcEntropy(self, data):
        num = data.shape[0] #rowcounts
        classCounts = {}
        for featVec in data:
            currentClass = featVec[-1] # last column shows class
            classCounts[currentClass] = classCounts.get(currentClass, 0) + 1
        entropy = 0.0
        for key in classCounts:
            prob = float(classCounts[key]) / num
            entropy -= prob * np.log2(prob)
        
        return entropy

    def _calcGini(self, data):
        num = data.shape[0]
        classCounts = {}
        for featVec in data:
            currentClass = featVec[-1]
            classCounts[currentClass] = classCounts.get(currentClass, 0) + 1
        gini = 1.0
        for key in classCounts:
            prob = float(classCounts[key]) / num
            gini -= prob ** 2
        return gini

    def _splitDataset(self, data, attrIndex, value):
        retIndex = []
        featVec = data[:, attrIndex]
        # new train dataset except selected attribute
        newData = data[:, [i for i in range(data.shape[1]) if i!=attrIndex]]
        for i in range(len(featVec)):
            if featVec[i] == value:
                retIndex.append(i)
        return newData[retIndex, :]

    def _chooseBestAttributeToSplit_ID3(self, data):
        numFeatures = data.shape[1] - 1
        parentEntropy = self._calcEntropy(data)
        bestInfoGain = 0.0
        bestAttributeIndex = -1
        # iterate each attributes
        for i in range(numFeatures):
            featList = data[:, i]
            valueSet = set(featList)
            newEntropy = 0.0
            for value in valueSet:
                subData = self._splitDataset(data, i, value)
                prob = float(len(subData)) / len(data)
                newEntropy += prob * self._calcEntropy(subData)
            infoGain = parentEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestAttributeIndex = i
        return bestAttributeIndex

    def _majorityCnt(self, labelList):
        '''投票器，列表中的众数'''
        labelCount = {}
        for vote in labelList:
            labelCount[vote] = labelCount.get(vote, 0) + 1
            sortedLabelCount = sorted(labelCount.items(), key=lambda x:x[1], reverse=True)
            return sortedLabelCount[0][0]

    def _chooseBestAttributeToSplit_C45(self, data):
        pass
    
    def _chooseBestAttributeToSplit_CART(self, data):
        pass
    
    def createTree(self, data, featList):
        # last column shows class label (target)
        labelList = [example[-1] for example in data]
        # class label is the same
        if labelList.count(labelList[0]) == len(labelList):
            return labelList[0]
        if len(data[0]) == 1:
            return self._majorityCnt(data)
        # data can be splited (using ID3)
        bestAttributeIndex = self._chooseBestAttributeToSplit_ID3(data)
        
        bestAttrStr = featList[bestAttributeIndex]
        featList = list(featList)
        featList.remove(bestAttrStr)
        featList = tuple(featList)

        newTree = {bestAttrStr: {}}
        featValueList = [example[bestAttributeIndex] for example in data]
        valueSet = set(featValueList)
        for value in valueSet:
            newData = self._splitDataset(data, bestAttributeIndex, value)
            newTree[bestAttrStr][value] = self.createTree(newData, featValueList)
        return newTree

    def fit(self, data):
        if isinstance(data, np.ndarray):
            pass
        else:
            try:
                data = np.array(data)
                print(data)
            except:
                raise TypeError("numpy.ndarray required for data")
        featList = tuple(['x_' +str(i) for i in range(len(data[0])-1)])
        self._tree = self.createTree(data, featList)
        return self

    def predict(self, X):
        if self._tree == None:
            raise TypeError("Model not fitted, call `fit` first")

        if isinstance(X, np.ndarray):
            pass
        else:
            try:
                X = np.array(X)
            except:
                raise TypeError("numpy.ndarray required for X")
        
        X = X[:]
        columnNames = ['x'+str(i) for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=columnNames)
        classification = []
        for record in df:
            classification.append(self._get_classification(self._tree, record))
        return classification
        
    def _get_classification(self, tree, record):
            """
            This function recursively traverses the decision tree and returns a
            classification for the given record.
            """
            if type(tree) == type("string"):
                return tree # leaf node
            else:
                # traverse the tree until leaf node
                attr = tree.keys()[0]
                subTree = tree[attr][record[attr]]
                return self._get_classification(subTree, record)
        

if __name__ == '__main__':
    dt = DecisionTree()
    data = [["a", "green", 1], ["a", "blue", 0], ["c", "blue", 1]]
    data = np.array(data)
    dt.fit(data)
    result = dt.predict([["a","green"]])