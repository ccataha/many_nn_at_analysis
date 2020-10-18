# Class to transformate data
import tools.data_tool as tfm
import model_results as model_results
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers


def defineModel():
    print("\n\n---  Creating LSTM Model                       ---")
    # Define new Model for rnn
    model = tf.keras.models.Sequential()
    # Adding model first layer with 41 features (42 - label feature)
    model.add(tf.keras.layers.LSTM(11))
    # Adding Hidden Layers
    # Adding Activation sigmoid on Hidden Layers
    model.add(tf.keras.layers.Dense(units=80,activation='sigmoid',name="dense_1"))
    model.add(tf.keras.layers.Dense(units=240,activation='sigmoid',name="dense_2"))
    model.add(tf.keras.layers.Dense(units=320,activation='sigmoid',name="dense_3"))
    # Adding output layer (normal(0) - anomaly(1))
    model.add(tf.keras.layers.Dense(units=7,activation='softmax',name="classification"))
    # Adding learning rate and metrics
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                loss=tf.keras.losses.mean_squared_error,
                metrics=['accuracy'])

    print("-------------------------------------------------")
    return model

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
    model_results.showSummaryClassification(TrainingDataY,results)

    print("\n\n---  Validate Results:                            ---")
    results = model.predict(validateDataX)
    model_results.showSummaryClassification(validateDataY,results)
    return model

def transformData(data):
    # Transform text to number
    print("---  Transform text to number                 ---")
    data = tfm.transformDataLabelClassification(data)
    # Split and remove columns
    print("---  Remove unused columns                    ---")
    kddCupY = data["label"]
    kddCupY = kddCupY.to_numpy()   
    kddCupY = keras.utils.to_categorical(kddCupY,7)
    kddCupX = data[["protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","lroot_shell","count","diff_srv_rate","dst_host_same_src_port_rate"]]
    print(data["protocol_type"])
    kddCupX = kddCupX.to_numpy()
    # Normalize data
    print("---  Normalize Colum Data                     ---")
    normalizeDataX = tfm.normalizeColumn(kddCupX)
    # Reshape
    print("---  Reshaping Data                           ---")
    normalizeDataX = normalizeDataX.reshape((normalizeDataX.shape[0],normalizeDataX.shape[1],-1))
    return normalizeDataX, kddCupY
