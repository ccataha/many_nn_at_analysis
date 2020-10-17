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


def transformData(data):
    # Transform text to number
    print("---  Transform text to number                 ---")
    data = tfm.transformDataClassification(data)
    data["label"]
    # Split and remove columns
    print("---  Remove unused columns                    ---")
    kddCupY = data["label"]
    # for row in kddCupY[:10]:
    #     if  kddCupY[row] == 0:
    #         kddCupY.drop([row])
    # print(kddCupY)
    kddCupY = kddCupY.to_numpy()   
    kddCupX = data
    kddCupX.pop("label")
    kddCupX = kddCupX.to_numpy()
    # Normalize data
    print("---  Normalize Colum Data                     ---")
    normalizeDataX = nml.normalizeColumn(kddCupX)
    kddCupY = kddCupY.reshape((kddCupY.shape[0],-1))
    normalizeDataY = nml.normalizeColumn(kddCupY)
    # Assing category data shape 
    normalizeDataY = keras.utils.to_categorical(normalizeDataY,7)
    # Reshape
    print("---  Reshaping Data                           ---")
    normalizeDataX = normalizeDataX.reshape((normalizeDataX.shape[0],normalizeDataX.shape[1],-1))
    return normalizeDataX, normalizeDataY
