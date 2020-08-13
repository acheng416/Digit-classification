import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import multiprocessing
import os
import itertools
import classifier
import sys
from contextlib import redirect_stdout
import cProfile
import perceptron
import shutil
import time
from line_profiler import LineProfiler

maxIter = 5
numTrials = 5


def runTrials(trial, trainingDigits, testingDigits, testingLabelsData , flattenedTestDigits, percent):
    

    testIndices = []
    guesses = []
    #Start timing bayes
    bayesStart = time.process_time()
    learningDigits = classifier.pickData(trainingDigits, percent)

    formattedDigits = classifier.getFormattedTraining(learningDigits)
            
    condProbCounters = classifier.getCondProbs(learningDigits)
    for i in range(0, len(testingDigits)):
        

        testIndices.append(i)
        #lp = LineProfiler()
        #lp_wrapper = lp(classifier.naiveBayes)
        #guesses.append(lp_wrapper(i, testingDigits, formattedDigits, percent, condProbCounters))
        #lp.print_stats()
        guesses.append(classifier.naiveBayes(i, testingDigits, formattedDigits, percent, condProbCounters))
    
    #End timing bayes
    bayesEnd = time.process_time()
    bayesTrainingTime = bayesEnd - bayesStart


    numCorrect = 0
    for tup in guesses:
        digitIndex = tup[0]
        guess = tup[1]
        if( int(guess) == int(testingLabelsData[digitIndex])):
            numCorrect+=1
    accBayes = (numCorrect/len(testingDigits)) * 100
    print("Accuracy for Naive Bayes classifier trial: " + str(trial) + " is: " + str(accBayes))

    #Begin perceptron training and prediction:
    numPixels = classifier.digitRowLen * classifier.digitColLen
    WVectors = perceptron.initializeWeightVecs(numPixels)

    
    #Start training time for perceptron
    startTime = time.process_time()
    for iteration in range(0, maxIter):
        #pool.starmap(perceptron.train, zip(itertools.repeat(WVectors), itertools.repeat(trainingDigits), itertools.repeat(percent)))
        perceptron.train(WVectors, trainingDigits, percent, iteration)

    #End training time
    endTime = time.process_time()
    perceptronTrainingTime = endTime - startTime 

    numCorrect = 0
    for k in range(0, len(flattenedTestDigits)):
	    testDigit = flattenedTestDigits[k]
	    guess = perceptron.predict(WVectors, testDigit)
	    if(guess == int(testingLabelsData[k])):
		    numCorrect+=1
    acc = (numCorrect/len(testingDigits)) * 100
    print("Accuracy for Perceptron trial: " + str(trial) + " is: " + str(acc))

        
    percentFolderPath = "{0:d} percent/".format(int(percent*100))


    with open("trainingData/" + percentFolderPath + "output{0:d}.txt".format(trial), "w+") as f:
            with redirect_stdout(f):
                print("%s %s %s %s" % ("Bayes Acc: ", str(accBayes) , "Bayes training time: ", str(bayesTrainingTime)))
                print("%s %s %s %s" % ("Percep Acc: ", str(acc), "Percep training time: ", str(perceptronTrainingTime)))


if __name__ == "__main__":
    

    #if os.path.exists("stats"):
        #shutil.rmtree("stats")

    #os.mkdir("stats")

    if os.path.exists("guesses"):
        shutil.rmtree("guesses")

    os.mkdir("guesses")

    if os.path.exists("trainingData"):
        shutil.rmtree("trainingData")
    os.mkdir("trainingData")


    multiprocessing.freeze_support()
    processorNum = os.cpu_count()
    pool = multiprocessing.Pool(processes = processorNum)

    trainingDigits = []
    classifier.loadTraining('digitdata/trainingimages', 'digitdata/traininglabels', trainingDigits)
    #print(len(digits))
    testingDigits = []
    classifier.loadTesting('digitdata/testimages', testingDigits)

    testingLabelsFile = open('digitdata/testlabels')
    testingLabelsData = testingLabelsFile.readlines()
    flattenedTestDigits = perceptron.get1D(testingDigits)

    trials = []
    for i in range(0, numTrials):
        trials.append(i)
    for percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        percentFolderPath = "{0:d} percent/".format(int(percent*100))
        os.mkdir("trainingData/" + percentFolderPath)
        pool.starmap(runTrials, zip(trials, itertools.repeat(trainingDigits), itertools.repeat(testingDigits), itertools.repeat(testingLabelsData) , itertools.repeat(flattenedTestDigits), itertools.repeat(percent)))
    

    pool.close()
    pool.join()
    #guesses = pool.starmap(classifier.naiveBayes, zip(testIndices,itertools.repeat(testingDigits), itertools.repeat(trainingDigits), itertools.repeat(percent)))
    
    
