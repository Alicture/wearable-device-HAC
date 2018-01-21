
# coding: utf-8
import numpy as np
from numpy import *
import operator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import os, fnmatch
from sklearn.preprocessing import scale
# 用于在jupyter中进行绘图
# get_ipython().run_line_magic('matplotlib', 'inline')
activity_df = pd.read_csv('dataset-har-PUC-Rio-ugulino.csv',sep=';',dtype={"user": str, "gender": int,"age":int,
                                                                         "how_tall_in_meters":float,"weight":int,
                                                                        "body_mass_index":float,
                                                                        "x1":int,"y1":int,"z1":int,
                                                                        "x2":int,"y2":int,"z2":int,
                                                                        "x3":int,"y3":int,"z3":int,
                                                                        "x4":int,"y4":int,"z4":int,
                                                                        "class":str})
# activity_df.head()

class_map={'sitting':1,'sittingdown':2,'standing':3,'standingup':4,'walking':5}
activity_df['class']=activity_df['class'].map(class_map)
# gender_map={'Woman':0,'Man':1}
# activity_df['gender']=activity_df['gender'].map(gender_map)

# activity_df.head()
X=activity_df[['gender','age','how_tall_in_meters','weight','body_mass_index','x1','y1','z1',
              'x2','y2','z2','x3','y3','z3','x4','y4','z4']]
y=activity_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,17))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(',')
        returnMat[index,:] = listFromLine[1:18]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector



def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals



# print(HARDataMat)
# print(HARLabels)
# print(normMat)


def Knn_Adv(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

HARDataMat,HARLabels = file2matrix('test2.txt')
# print(HARDataMat)
normMat, ranges, minVals = autoNorm(HARDataMat)
hoRatio = 0.25
m = normMat.shape[0]
numTestVecs = int(m*hoRatio)
errorCount = 0.0
for i in range(numTestVecs):
    classifierResult = Knn_Adv(normMat[i,:],normMat[numTestVecs:m,:],HARLabels[numTestVecs:m],3)
    print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, HARLabels[i]))
    if (classifierResult != HARLabels[i]): errorCount += 1.0
print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))
print (errorCount)

