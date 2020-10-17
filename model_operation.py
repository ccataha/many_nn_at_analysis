# Class to transformate data
import tools.transformation as tfm
import tools.normalization as nml
import model_results as model_results
import tensorflow as tf
from tensorflow import keras
import numpy as np

def trainModel(epochs,  model, trainingData, validateData):
    print("---  Transform and Normalize Training Data    ---")
    TrainingDataX, TrainingDataY = transformData(trainingData)
    # Training Model
    print("---  Training model    ---")
    model.fit(TrainingDataX,TrainingDataY,epochs=epochs,batch_size=25,shuffle=True)
    # validation_data=(validateDataX,validateDataY))
    results = model.predict(TrainingDataX)

    print("\n\n---  Results:            ---")
    model_results.showSummary(TrainingDataY,results)
    return model

def validateModel(model, validateData):
    print("---  Transform and Normalize Validate Data    ---")
    validateDataX, ValidateDataY = transformData(validateData)
    # Validate Model
    print("---  Validate model    ---")
    results = model.predict(validateDataX)
    print("\n\n---  Results:            ---")
    model_results.showSummary(ValidateDataY,results)
    return model

def transformData(data):
    # Transform text to number
    print("---   Transform text to number        ---")
    data = tfm.transformData(data)
    data["label"]
    # Split and remove columns
    print("---   Remove unused columns           ---")
    kddCupY = data["label"]
    kddCupY = kddCupY.to_numpy()   
    # Assing category data shape 
    kddCupY = keras.utils.to_categorical(kddCupY,2)
    kddCupX = data
    kddCupX.pop("label")
    kddCupX = kddCupX.to_numpy()
    # Normalize data
    print("---  Normalize Colum Data            ---")
    normalizeDataX = nml.normalizeColumn(kddCupX)
    # Reshape
    print("---  Reshaping Data                  ---")
    normalizeDataX = normalizeDataX.reshape((normalizeDataX.shape[0],normalizeDataX.shape[1],-1))
    return normalizeDataX, kddCupY
