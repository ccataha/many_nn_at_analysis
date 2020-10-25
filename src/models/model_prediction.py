# Class to transformate data
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers

import tools.data_tool as tfm
import src.result.model_results as model_results
import src.result.model_save as model_save



def defineModel():
    print("\n\n---  Creating RNN Model                       ---")
    # Define new Model for rnn
    model = tf.keras.models.Sequential()
    # Adding model first layer with 41 features (42 - label feature)
    model.add(tf.keras.layers.SimpleRNN(41))
    # Adding Hidden Layers
    # Adding Activation sigmoid on Hidden Layers
    model.add(tf.keras.layers.Dense(units=82,activation='sigmoid',name="dense_1"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units=164,activation='sigmoid',name="dense_2"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units=248,activation='sigmoid',name="dense_3"))
    # Adding output layer (normal(0) - anomaly(1))
    model.add(tf.keras.layers.Dense(units=2,activation=tf.nn.softmax,name="predictions"))
    # Adding learning rate and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.mean_squared_error,
                metrics=['accuracy'])

    print("-------------------------------------------------")
    return model

def transformData(data):
    # Transform text to number
    print("---  Transform text to number                 ---")
    data = tfm.transformDataLabel(data)
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
    normalizeDataX = tfm.normalizeColumn(kddCupX)
    # Reshape
    print("---  Reshaping Data                           ---")
    normalizeDataX = normalizeDataX.reshape((normalizeDataX.shape[0],normalizeDataX.shape[1],-1))
    return normalizeDataX, kddCupY

def trainModel(epochs,  model, trainingData, validateData):
    # Training Model
    print("\n\n---  Training model                           ---")
    print("---  Transform and Normalize Training Data    ---")
    print("---  Training                                 ---")
    TrainingDataX, TrainingDataY = transformData(trainingData)
    validateDataX, validateDataY = transformData(validateData)
    model.fit(TrainingDataX,TrainingDataY,epochs=epochs,batch_size=25,shuffle=True,
    validation_data=(validateDataX,validateDataY))
    model_save.saveModel(model,'prediction_model')    
    predictModel(model, TrainingDataX, TrainingDataY)
    validateModel(model,validateDataX, validateDataY)
    return model

# PREDICT, VALIDATE, EVALUATE
def predictModel(model, TrainingDataX, TrainingDataY):
    print("\n\n---  Predict Results:                            ---")
    results = model.predict(TrainingDataX)
    print(results)
    print("-------------------------------------------------")
    return results

def validateModel(model, validateDataX, validateDataY):
    print("\n\n---  Validate Results:                            ---")
    results = model.predict(validateDataX)
    model_results.showSummary(validateDataY,results)
    return results

def predictFullModel(model, data):
    print("\n\n---  Predict Results Predition:                   ---")
    data = tfm.transformDataLabel(data)
    kddCupX = data
    if 'label' in kddCupX.columns:
        kddCupX.pop("label")
    
    kddCupX = kddCupX.to_numpy()
    normalizeDataX = tfm.normalizeColumn(kddCupX)
    normalizeDataX = normalizeDataX.reshape((normalizeDataX.shape[0],normalizeDataX.shape[1],-1))
    results = model.predict(normalizeDataX)   
    predictedColumn = np.argmax(results, axis=1)
    return predictedColumn

def evaluateModel(model, testData):
    print("\n\n---  Evaluate model                           ---")
    print("---  Transform and Normalize Evaluate Data       ---")
    testDataX, testDataY = transformData(testData)
    # Validate Model
    print("\n\n---  Evaluate Results:                            ---")
    results = model.evaluate(testDataX,testDataY)
    print(results)
    return results

