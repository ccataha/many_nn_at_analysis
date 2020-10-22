import model_data as data_processing
import model_prediction as model_prediction
import model_classification as model_classification
import model_save as model_save

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers

def askOperation():
    confirm = input("\n\nWhat kind of NN you want to implement RNN - Prediction [1] or LSTM - Classification [2]  ?")
    if confirm != '1' and confirm != '2':
        print("\n\nInvalid Option. Please Enter a Valid Option.")
        askOperation()
    else:
        if confirm == '1':
            preddictionModel()
        else:
            classificationModel()

# PREDICTION
def preddictionModel():
    confirm = input("\n\nDo you want to Train[1], Evaluate[2]?")
    if confirm != '1' and confirm != '2':
        print("\n\nInvalid Option. Please Enter a Valid Option.")
        askOperation()
    else:
        if confirm == '1':
           predictTrainOption()
        else:
            predictEvaluateOption()

def predictTrainOption():
     # Get Data
    trainingData, validateData, testData = data_processing.readData("kddcup99_csv.csv")
    # Create Model Prediction
    model = model_prediction.defineModel()
    # Train and Validate Model
    model = model_prediction.trainModel(3, model, trainingData, validateData)

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
    confirm = input("\n\nDo you want to Train[1], Evaluate[2]?")
    if confirm != '1' and confirm != '2':
        print("\n\nInvalid Option. Please Enter a Valid Option.")
        askOperation()
    else:
        if confirm == '1':
           classificationTrainOption()
        else:
            classificationEvaluateOption()

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


# Program header
print("---  Starting Neural Network                  ---")
print("-------------------------------------------------")
askOperation()