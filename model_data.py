# Library to read excel data
import pandas as pd 
import numpy as np

def readData(file):

    # Read csv data
    print("\n\n---  Reading csv data                         ---")
    dirpath = "resources/"
    filename = file
    kddCup = pd.read_csv(dirpath+filename)

    # Shuffle data
    print("---  Shuffle data                             ---")
    kddCup = kddCup.sample(frac=1)

    # Split data in 70% 15% 15%
    print("---  Split data in train, validate and test   ---")
    trainingData, validateData, testData = np.split(kddCup.sample(frac=1), [int(.7*len(kddCup)), int(.85*len(kddCup))])

    print("---  TrainingData 70% length = " + str(len(trainingData))+"          ---")
    print("---  ValidateData 15% lenght = " + str(len(validateData))+"          ---")
    print("---  TestData 15% length     = " + str(len(testData))+"          ---")
    print("-------------------------------------------------")

    # Shuffle data
    # print("---  Shuffle data                             ---")
    # trainingData = trainingData.sample(frac=1)
    # validateData = validateData.sample(frac=1)
    # testData = testData.sample(frac=1)
    return trainingData, validateData, testData
