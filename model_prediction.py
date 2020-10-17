# Class to transformate data
import tools.transformation as tfm
import tools.normalization as nml
import model_results as model_results
import tensorflow as tf
from tensorflow import keras
import numpy as np

def trainModel(epochs,  model, trainingData, validateData):
    # Training Model
    print("\n\n---  Training model                           ---")
    print("---  Transform and Normalize Training Data    ---")
    print("---  Training                                 ---")
    TrainingDataX, TrainingDataY = transformData(trainingData)
    validateDataX, validateDataY = transformData(validateData)
    model.fit(TrainingDataX,TrainingDataY,epochs=epochs,batch_size=25,shuffle=True,
    validation_data=(validateDataX,validateDataY))
    
    print("\n\n---  Training Results:                            ---")
    results = model.predict(TrainingDataX)
    model_results.showSummary(TrainingDataY,results)

    print("\n\n---  Validate Results:                            ---")
    results = model.predict(validateDataX)
    model_results.showSummary(validateDataY,results)
    return model

def evaluateModel(model, testData):
    print("\n\n---  Evaluate model                           ---")
    print("---  Transform and Normalize Evaluate Data       ---")
    testDataX, testDataY = transformData(testData)
    # Validate Model
    print("\n\n---  Evaluate Results:                            ---")
    results = model.evaluate(testDataX)
    # model_results.showSummary(testDataY,results)
    print(results)

    print("-------------------------------------------------")
    # Validate Model
    print("\n\n---  Predict                                    ---")
    results = model.predict(testDataX[:10])
    print("---  Results:                               ---")
    model_results.showSummary(testDataY,results)

    print("-------------------------------------------------")
    return model

def transformData(data):
    # Transform text to number
    print("---  Transform text to number                 ---")
    data = tfm.transformData(data)
    data["label"]
    # Split and remove columns
    print("---  Remove unused columns                    ---")
    kddCupY = data["label"]
    kddCupY = kddCupY.to_numpy()   
    # Assing category data shape 
    kddCupY = keras.utils.to_categorical(kddCupY,2)
    kddCupX = data
    kddCupX.pop("label")
    kddCupX = kddCupX.to_numpy()
    # Normalize data
    print("---  Normalize Colum Data                     ---")
    normalizeDataX = nml.normalizeColumn(kddCupX)
    # Reshape
    print("---  Reshaping Data                           ---")
    normalizeDataX = normalizeDataX.reshape((normalizeDataX.shape[0],normalizeDataX.shape[1],-1))
    return normalizeDataX, kddCupY
