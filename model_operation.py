# Class to transformate data
import controller.transformation as trans
import controller.normalization as norm
import tensorflow as tf
from tensorflow import keras
import model_results as mrs
import numpy as np

def transformData(data):
    # Transform text to number
    print("Transform text to number")
    data = trans.transformData(data)
    data["label"]
    # Split and remove columns
    print("Remove unused columns")
    kddCupY = data["label"]
    kddCupY = kddCupY.to_numpy()    
    kddCupY = keras.utils.to_categorical(kddCupY,2)
    kddCupX = data
    kddCupX.pop("label")
    kddCupX = kddCupX.to_numpy()
    # Normalize data
    print("Normalize Colum Data")
    normalizeTrainingDataX = norm.normalizeColumn(kddCupX)
    # Reshape
    print("Reshaping Data")
    normalizeTrainingDataX = normalizeTrainingDataX.reshape((normalizeTrainingDataX.shape[0],normalizeTrainingDataX.shape[1],-1))
    return normalizeTrainingDataX, kddCupY

def trainModel(epochs,  model, trainingData, validateData):
    print("Transform and Normalize Training")
    normalizeTrainingDataX, arrayTrainingDataY = transformData(trainingData)
    print("Transform and Normalize Validation")
    validateDataX, validateDataY = transformData(validateData)

    # Train Model
    print("\n\nTraining model with new data")
    model.fit(normalizeTrainingDataX,arrayTrainingDataY,epochs=epochs,batch_size=25,shuffle=True,
    validation_data=(validateDataX,validateDataY))
    
    results = model.predict(normalizeTrainingDataX)

    print("\n\nResults:")
    mrs.showSummary(arrayTrainingDataY,results)
    return model

def validateModel(model, validateData):
    normalizeValidateDataX, arrayValidateDataY = transformData(validateData)
    print("\n\nValidate Model with new data")
    results = model.predict(normalizeValidateDataX)
    print("\n\nResults:")
    mrs.showSummary(results,arrayValidateDataY)
    return model
