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
import classifier
import time
import random

def pickData(trainingSet, percent):
    randGen = np.random.default_rng()
    goal = int(len(trainingSet) * percent)
    possibleIndices = np.arange(start=0, stop=len(trainingSet), step=1, dtype=int)
    indices = randGen.choice(possibleIndices, size=goal, replace=False)

    ls = []
    for index in indices:
        ls.append(trainingSet[index])

    return ls

def pickDataA(trainingSet, percent):
	if percent == 1:
		return trainingSet
	ls = []
	goal = int(len(trainingSet) * percent)
	count = 0
	random.seed(time.time())
	while(count < goal):
		k = random.randint(0, len(trainingSet) - 1)
		if not trainingSet[k] in ls:
			ls.append(trainingSet[k])
			count = count + 1
	return ls

def get1D(trainingDigits):
    flattenedDigits = []
    for digit in trainingDigits:
        currentFlatDigit = []
        for i in range(0, len(digit.Mat)):
            for j in range(0, len(digit.Mat[i])):
                currentPixelVal = digit.Mat[i][j]
                currentFlatDigit.append(currentPixelVal)
        flattenedDigits.append(currentFlatDigit)
    return flattenedDigits

def initializeWeights(WVector, size):
	random.seed(time.process_time() + 300)
	for i in range(0, size):
		#randNum = random.uniform(-1.0, 1.0)
		#WVector.append(randNum)
		WVector.append(0.0)
	return WVector

def initializeWeightVecs(numPixels):
	WVectors = {}
	for label in range(0, 10):
		currentWVec = []
		currentWVec = initializeWeights(currentWVec, numPixels + 1)
		WVectors[label] = currentWVec
	return WVectors 

def weightedSumsNP(digit, newWVectors):
	weightedSums = []
	numPixels = len(digit)
	for i in range(0, 10):
		WVector = newWVectors[i]
		bias = WVector[0] 
		del WVector[0]
		#print("Digit len: " + str(len(digit)))
		#print("Weight len: " + str(len(WVector)))
		resultMat = np.dot(digit, WVector)
		currentSum = bias + np.sum(resultMat)
		weightedSums.append(currentSum)
		WVector.insert(0, bias)
		
	#print(weightedSums)
	#print("Returning")
	return weightedSums

def weightedSums(digit, WVectors):
	weightedSums = []
	numPixels = len(digit)
	for i in range(0, len(WVectors)):
		WVector = WVectors.get(i)
		#print(len(WVector))
		#print(WVector)
		currentSum = WVector[0]
		#print("currentSum: " + str(currentSum))
		for j in range(0, numPixels):
			currentSum = currentSum + (digit[j] * WVector[j+1])
		weightedSums.append(currentSum)
	#print(weightedSums
	return weightedSums

def argMax(weightedSums):
	maxVal = -(sys.maxsize)
	argMax = -1
	for i in range(0, len(weightedSums)):
		if(weightedSums[i] >= maxVal):
			maxVal = weightedSums[i]
			argMax = i
	return argMax

def updateWeights(WVector, updateType, digit):
	if(updateType == "penalize"):
		for i in range(1, len(WVector)):
			WVector[i] = WVector[i] - digit[i-1]
		return WVector 
	elif(updateType == "reward"):
		for i in range(1, len(WVector)):
			WVector[i] = WVector[i] + digit[i-1]
		return WVector


def train(WVectors, trainingDigits, percent, iteration):
	learningDigits = pickData(trainingDigits, percent)
	flattenedDigits = get1D(learningDigits)
	newWVectors = convertTo2DMat(WVectors)

	for i in range(0, len(flattenedDigits)):
		digit = flattenedDigits[i]
		trueLabel = learningDigits[i].Val
		#wSums = weightedSums(digit, WVectors)
		wSums = weightedSumsNP(digit, WVectors)
		guess = argMax(wSums)
		
		#print("guess is: " + str(guess))
		if(int(trueLabel) == guess):
			continue
		elif(int(trueLabel) != guess):
			wGuess = WVectors.get(guess)
			wLabel = WVectors.get(int(trueLabel))
			#print(trueLabel)
			#Update (penalize) guess weights
			wGuess = updateWeights(wGuess, "penalize", digit)
			#Update (reward) trueLabel weights
			wLabel = updateWeights(wLabel, "reward", digit)
	#np.savetxt("guesses/weights{0:d}.txt".format(iteration), newWVectors)

def predict(WVectors, testDigit):
	scores = weightedSums(testDigit, WVectors)
	guess = argMax(scores)
	return guess

def convertTo2DMat(WVectors):
	resultMat = []
	for label in range(0, len(WVectors)):
		currentVector = []
		WVector = WVectors.get(label)
		for pixelVal in WVector:
			currentVector.append(pixelVal)
			
		resultMat.append(currentVector)
    
	return resultMat
