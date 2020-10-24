import model_data as data_processing
import model_prediction as model_prediction
import model_classification as model_classification
import model_save as model_save

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import pandas as pd 

def askOperation():

    print("\n\nWhat kind of NN you want to implement")
    print("\n\n--> RNN - Prediction [1]")
    print("--> LSTM - Classification [2]")
    print("--> Prediction with Classification [3]")
    confirm = input("\n\nSelect option: ")
    if confirm != '1' and confirm != '2' and confirm != '3':
        print("\n\nInvalid Option. Please Enter a Valid Option.")
        askOperation()
    else:
        if confirm == '1':
            preddictionModel()
        else:
            if confirm == '2':
                classificationModel()
            else:
                fullModel()

# PREDICTION
def preddictionModel():
    confirm = input("\n\nDo you want to Train[1], Exit[2]?")
    if confirm != '1' and confirm != '2':
        print("\n\nInvalid Option. Please Enter a Valid Option.")
        askOperation()
    else:
        if confirm == '1':
           predictTrainOption()
        else:
            # predictEvaluateOption()
            print('Exit')

def predictTrainOption():
     # Get Data
    trainingData, validateData, testData = data_processing.readData("kddcup99_csv.csv")
    # Create Model Prediction
    model = model_prediction.defineModel()
    # Train and Validate Model
    model = model_prediction.trainModel(5, model, trainingData, validateData)

def predictEvaluateOption():
    model_load = model_save.loadModel('prediction_model')
    if model_load == None:
        print('Please train first your model')
        askOperation()
    else:
        trainingData, validateData, testData = data_processing.readData("kddcup99_csv_balance.csv")
        model_classification.evaluateModel(model_load,testData)

# CLASSIFICATION
def classificationModel():
    confirm = input("\n\nDo you want to Train[1], Exit[2]?")
    if confirm != '1' and confirm != '2':
        print("\n\nInvalid Option. Please Enter a Valid Option.")
        askOperation()
    else:
        if confirm == '1':
           classificationTrainOption()
        else:
            # classificationEvaluateOption()
            print('Exit')

def classificationTrainOption():
    # Get Data
    trainingData, validateData, testData = data_processing.readData("kddcup99_csv_balance.csv")
    # Create Model Classification
    model = model_classification.defineModel()
    # Train and Validate Model
    model = model_classification.trainModel(3, model, trainingData, validateData)

def classificationEvaluateOption():
    model_load = model_save.loadModel('classification_model')
    if model_load == None:
        print('Please train first your model')
        askOperation()
    else:        
        trainingData, validateData, testData = data_processing.readData("kddcup99_csv_balance.csv")
        model_classification.evaluateModel(model_load,trainingData)

# PREDICTION AND CLASSIFICATION
def fullModel():
    print("\n\n---  Training Models                          ---")
    model_Pload = model_save.loadModel('prediction_model')
    model_Cload = model_save.loadModel('classification_model')
    # Read csv data

    print("\n\n---  Reading csv data                         ---")
    dirpath = "resources/"
    filename = input("Enter File Name:  ")
    dataFile = pd.read_csv(dirpath+filename)
    predictionData = dataFile.copy()

    predictColum = model_prediction.predictFullModel(model_Pload,predictionData)
    dataFile['predicted'] = predictColum
    dataFile = dataFile[dataFile.predicted != '0'] 
    dataFile.pop('predicted')
    resultColum = model_classification.predictFullModel(model_Cload,dataFile)
    return resultColum

# Program header
print("---  Starting Neural Network                  ---")
print("-------------------------------------------------")
askOperation()


    # # recorrerla y eliminar todos los normales
    # for i in range(0, len(predictColum)):
    #     # predictColum[i] = valor de la predeccion 0-1
    #     # if(predictColum[i] = '0')
    # remove normal 