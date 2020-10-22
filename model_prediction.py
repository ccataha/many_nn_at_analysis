# Class to transformate data
import tools.data_tool as tfm
import model_results as model_results
import model_save as model_save
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers


def defineModel():
    print("\n\n---  Creating RNN Model                       ---")
    # Define new Model for rnn
    model = tf.keras.models.Sequential()
    # Adding model first layer with 41 features (42 - label feature)
    model.add(tf.keras.layers.SimpleRNN(41))
    # Adding Hidden Layers
    # Adding Activation sigmoid on Hidden Layers
    model.add(tf.keras.layers.Dense(units=80,activation='sigmoid',name="dense_1"))
    model.add(tf.keras.layers.Dense(units=160,activation='sigmoid',name="dense_2"))
    model.add(tf.keras.layers.Dense(units=240,activation='sigmoid',name="dense_3"))
    # Adding output layer (normal(0) - anomaly(1))
    model.add(tf.keras.layers.Dense(units=2,activation='softmax',name="predictions"))
    # Adding learning rate and metrics
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
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
