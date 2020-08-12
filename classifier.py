import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import multiprocessing
import os
import itertools
import sys
from contextlib import redirect_stdout
import cProfile
import math
import random
import time
from line_profiler import LineProfiler



digitColLen = 29
digitRowLen = 20


class digit:
    def __init__(self):
        self.Mat = [ [0]*digitColLen for row in range(digitRowLen)]
        self.Val = -1

def pickData(trainingSet, percent):
    randGen = np.random.default_rng()
    goal = int(len(trainingSet) * percent)
    possibleIndices = np.arange(start=0, stop=len(trainingSet), step=1, dtype=int)
    indices = randGen.choice(possibleIndices, size=goal, replace=False)

    ls = []
    for index in indices:
        ls.append(trainingSet[index])

    return ls


def printDigitMat(digit):
    for i in range(0, digitRowLen):
        for j in range(0, digitColLen):
            print(str(digit.Mat[i][j]) + "", end="")
        print()

def loadRow(digit, rowIndex, line):
        #print(line, end="")
        #print("Loading row: " + str(rowIndex))
        for j in range(0, digitColLen):
            if(line[j] == '+'):
                #print("AAA")
                digit.Mat[rowIndex][j] = 1
            elif(line[j] == '#'):
                digit.Mat[rowIndex][j] = 2
            else:
                #print("[loadRow]" + line[j], end="")
                digit.Mat[rowIndex][j] = 0

def loadTraining(trainingpath, labelpath, trainingDigits):
    trainingFile = open(trainingpath)
    trainingData = trainingFile.readlines()
    trainingLabelFile = open(labelpath)
    trainingLabelData = trainingLabelFile.readlines()
    numDigitsParsed = 0

    numLines = 0
    for line in trainingData:
        numLines+=1

    i = 0
    while(i != numLines):
        currentLine = trainingData[i]
        #print(currentLine)

        if(('+' not in currentLine) and ('#' not in currentLine)): #Empty line -> skip
            i+=1; continue

        digitStart = i
        #print("digit: " + str(numDigitsParsed+1) + " 's start is: " + str(digitStart))
        rowIndex = 0
        newDigit = digit()
        newDigit.Val = trainingLabelData[numDigitsParsed]
        trainingDigits.append(newDigit)
        numDigitsParsed+=1
        while(i<digitStart+digitRowLen):
            currentLine = trainingData[i]
            #print(currentLine, end="")
            #Load line into ith row of digit object's matrix as binary
            loadRow(newDigit, rowIndex, currentLine)
            i+=1
            rowIndex+=1
        

def calcPriors(trainingDigits): #Compute P(C), where C is {0,1,2,...,9} for digits and {face, not-face} for faces
    priorVec = []
    for label in range(0, 10):
        numDigit = numLabel(label, trainingDigits)
        currentPrior = (numDigit/len(trainingDigits))
        priorVec.append(currentPrior)
    return priorVec

def numLabel(label, data):
    numLabel = 0
    for datum in data.get(label):
        if(int(datum.Val) == label):
            numLabel+=1
    return numLabel



def calcCondProb(pixelVal, pixelRow, pixelCol, label, trainingDigits): #Calculates P(Fi = fi | C = c), where Fi = pixel i, fi = pixel Fi's value
    matches = [1 for digit in trainingDigits.get(label) if (digit.Mat[pixelRow][pixelCol] == pixelVal)]
    return (len(matches)/len(trainingDigits.get(label)))


def calcLikelihoods(testDigit, trainingDigits):
    likelihoods = []
    #printDigitMat(testDigit)
    for label in range(0, 10):
        likelihood = 1.0

        for i in range(0, digitRowLen):
            for j in range(0, digitColLen):
                pixelVal = testDigit.Mat[i][j]
                
                conditionalProb = calcCondProb(pixelVal, i, j, label, trainingDigits)
                #print("CondProb is: " + str(conditionalProb) + " for pixel: (" + str(j) + " , " + str(i) + ")")
                likelihood = likelihood* conditionalProb
                if(likelihood == 0.0):
                    break
        #print("likelihood for a " + str(label) + " : " + str(likelihood))
        likelihoods.append(likelihood)
    return likelihoods

def getFormattedTraining(trainingDigits):
    formattedData = {}
    for label in range(0, 10):
        currentLabelDigits = []
        for digit in trainingDigits:
            if(int(digit.Val) == label):
                currentLabelDigits.append(digit)
        formattedData[label] = currentLabelDigits
    return formattedData
 


def calcPosteriors(testDigit, trainingDigits):
    #lp = LineProfiler()
    #lp_wrapper = lp(calcLikelihoods)
    #likelihoods = lp_wrapper(testDigit, trainingDigits)
    #lp.print_stats()


    likelihoods = calcLikelihoods(testDigit, trainingDigits)
    priors = calcPriors(trainingDigits)
    posteriors = [(likelihoods[label] * priors[label]) for label in range(0, 10)]
    return posteriors

def naiveBayes(testDigitNum, testDigits, trainingDigits, percent):
    #Add in selecting data
    with open("guesses/output{0:d}.txt".format(testDigitNum), "w+") as f:
        with redirect_stdout(f):
            testDigit = testDigits[testDigitNum]
            #print("Test digit: " + str(testDigitNum+1))
            
            learningDigits = pickData(trainingDigits, percent)

            formattedDigits = getFormattedTraining(learningDigits)
            
            posteriors = calcPosteriors(testDigit, formattedDigits)
            argMax = -1
            maxLabel = -1
            for label in range (0, 10):
                if(posteriors[label] >= argMax):
                    argMax = posteriors[label]
                    maxLabel = label
            #print("Guess is : " + str(maxLabel))
            tup = testDigitNum, maxLabel
            return tup

def loadTesting(testingpath, testingDigits):
    testingFile = open(testingpath)
    testingData = testingFile.readlines()
    numDigitsParsed = 0

    numLines = 0
    for line in testingData:
        numLines+=1

    i = 0
    while(i != numLines):
        currentLine = testingData[i]
        #print(currentLine)

        if(('+' not in currentLine) and ('#' not in currentLine)): #Empty line -> skip
            i+=1; continue

        digitStart = i
        #print("digit: " + str(numDigitsParsed+1) + " 's start is: " + str(digitStart))
        rowIndex = 0
        newDigit = digit()
        testingDigits.append(newDigit)
        numDigitsParsed+=1
        while(i<digitStart+digitRowLen):
            currentLine = testingData[i]
            #print(currentLine, end="")
            #Load line into ith row of digit object's matrix as binary
            loadRow(newDigit, rowIndex, currentLine)
            i+=1
            rowIndex+=1

