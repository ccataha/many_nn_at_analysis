# Library to read excel data
import pandas as pd 
import numpy as np
# Operations
import model_definition as model_df
import model_operation as model_op

def readData():

    # Read csv data
    print("---  Reading csv data            ---")
    dirpath = "resources/"
    filename = "kddcup99_csv.csv"
    kddCup = pd.read_csv(dirpath+filename)

    # Shuffle data
    print("---  Shuffle data                 ---")
    kddCup = kddCup.sample(frac=1)

    # Split data in 70% 15% 15%
    print("---  Split data in train, validate and test  ---")
    trainingData, validateData, testData = np.split(kddCup.sample(frac=1), [int(.7*len(kddCup)), int(.85*len(kddCup))])

    print("---  TrainingData 70% length = " + str(len(trainingData))+" ---")
    print("---  ValidateData 15% lenght = " + str(len(validateData))+" ---")
    print("---  TestData 15% length     = " + str(len(testData))+" ---")
    return trainingData, validateData, testData
