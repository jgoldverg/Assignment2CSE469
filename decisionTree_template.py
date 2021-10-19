import collections

import treeplot
import pandas as pd
import numpy as np


def loadDataSet(filepath):
    '''
    Returns
    -----------------
    data: 2-D list
        each row is the feature and label of one instance
    featNames: 1-D list
        feature names
    '''
    data=[]
    featNames = None
    fr = open(filepath)
    for (i,line) in enumerate(fr.readlines()):
        array=line.strip().split(',')
        if i == 0:
            featNames = array[:-1]
        else:
            data.append(array)
    return data, featNames


def splitData(dataSet, axis, value):
    '''
    Split the dataset based on the given axis and feature value

    Parameters
    -----------------
    dataSet: 2-D list
        [n_sampels, m_features + 1]
        the last column is class label
    axis: int 
        index of which feature to split on
    value: string
        the feature value to split on

    Returns
    ------------------
    subset: 2-D list 
        the subset of data by selecting the instances that have the given feature value
        and removing the given feature columns
    '''
    subset = []
    for instance in dataSet:
        if instance[axis] == value:    # if contains the given feature value
            reducedVec = instance[:axis] + instance[axis+1:] # remove the given axis
            subset.append(reducedVec)
    return subset



'''
    tuple form is: 
'''
def chooseBestFeature(dataSet):
    '''
    choose best feature to split based on Gini index
    Parameters
    -----------------
    dataSet: 2-D list
        [n_sampels, m_features + 1]
        the last column is class label

    Returns
    ------------------
    bestFeatId: int
        index of the best feature
    '''
    #TODO
    gain_list = []
    dataFrame = pd.DataFrame(dataSet)
    classLabeCounts = dataFrame[dataFrame.columns[-1]].value_counts() #Yes and NO labels of the whole dataset
    bigGini = giniMath(dataFrame.shape[0], classLabeCounts) #This is the big gini of the entire dataset
    for i in range(len(dataFrame.columns)-1): #looping through all the m feature columns and not label column
        uniqueValuesInFeature = dataFrame[i].unique() #unique features of the column we are at
        featureCounts = dataFrame[i].value_counts()
        giniList = [] #list that will store little ginis
        fractionList = [] #list that will store the whole fractions
        for label in uniqueValuesInFeature:
            df = pd.DataFrame(splitData(dataSet, i, label))
            subsetCount = df[df.columns[-1]].value_counts() # last column yes and no counts. The biggest one first
            giniList.append(giniMath(df.shape[0], subsetCount))
            fractionList.append(featureCounts[label]/dataFrame.shape[0])
        gain_list.append(gainMath(bigGini, fractionList, giniList))
    print("The gain list: ", gain_list)
    if len(gain_list) == 0: return 0
    return gain_list.index(max(gain_list))

def gainMath(bigGini, fractionList, lilGiniList):
    print("Big Gini: ", bigGini, " fraction list is: ", fractionList, " lilGiniList is: ", lilGiniList)
    giniTotal = 0
    for i in range(len(fractionList)):
        frac = fractionList[i]
        lilGini = lilGiniList[i]
        giniTotal += (frac * lilGini)
    return bigGini - giniTotal

def giniMath(n,classLabelCount):
    total =1
    for key,value in enumerate(classLabelCount):
         total-=((value/n)**2)
    return total

def stopCriteria(dataSet):
    '''
    Criteria to stop splitting: 
    1) if all the classe labels are the same, then return the class label;
    2) if there are no more features to split, then return the majority label of the subset.

    Parameters
    -----------------
    dataSet: 2-D list
        [n_sampels, m_features + 1]
        the last column is class label

    Returns
    ------------------
    assignedLabel: string
        if satisfying stop criteria, assignedLabel is the assigned class label;
        else, assignedLabel is None 
    '''
    assignedLabel = None
    # TODO
    df = pd.DataFrame(dataSet)
    lastColumn = df[df.columns[-1]]
    counts = lastColumn.value_counts() #get all unique values and occurence counts
    countsList = counts.index.tolist() #convert the unique values to sorted list by the counts
    if len(df.columns) == 1: return countsList[0]
    if(len(counts) < 2): return countsList[0] #if there is only one count return that
    return assignedLabel

def buildTree(dataSet, featNames):
    '''
    Build the decision tree

    Parameters
    -----------------
    dataSet: 2-D list
        [n'_sampels, m'_features + 1]
        the last column is class label

    Returns
    ------------------
        myTree: nested dictionary
    '''
    assignedLabel = stopCriteria(dataSet)
    if assignedLabel:
        return assignedLabel
    bestFeatId = chooseBestFeature(dataSet)
    print("Best feature idx is ", bestFeatId, " which has value ", featNames[bestFeatId])
    bestFeatName = featNames[bestFeatId]

    myTree = {bestFeatName:{}}
    subFeatName = featNames[:]
    del(subFeatName[bestFeatId])
    featValues = [d[bestFeatId] for d in dataSet]
    uniqueVals = list(set(featValues))
    for value in uniqueVals:
        myTree[bestFeatName][value] = buildTree(splitData(dataSet, bestFeatId, value), subFeatName)
    
    return myTree



if __name__ == "__main__":
    data, featNames = loadDataSet('car.csv')
    dtTree = buildTree(data, featNames)
    print (dtTree)
    treeplot.createPlot(dtTree)