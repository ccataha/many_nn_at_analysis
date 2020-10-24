# Class to transformate data
import tools.data_tool as tfm
import model_results as model_results
import model_save as model_save
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import time


def defineModel():
    print("\n\n---  Creating LSTM Model                       ---")
    # Define new Model for rnn
    model = tf.keras.models.Sequential()
    # Adding model first layer with 41 features (42 - label feature)
    model.add(tf.keras.layers.LSTM(11))
    # Adding Hidden Layers
    # Adding Activation sigmoid on Hidden Layers
    model.add(tf.keras.layers.Dense(units=22,activation=tf.nn.softsign,name="dense_1"))
    model.add(tf.keras.layers.Dense(units=44,activation=tf.nn.softsign,name="dense_2"))
    model.add(tf.keras.layers.Dense(units=88,activation=tf.nn.softsign,name="dense_3"))
    model.add(tf.keras.layers.Dense(units=176,activation=tf.nn.softsign,name="dense_4"))
    model.add(tf.keras.layers.Dense(units=264,activation=tf.nn.softsign,name="dense_5"))
    # Adding output layer (normal(0) - anomaly(1))
    model.add(tf.keras.layers.Dense(units=7,activation=tf.nn.softsign.softmax,name="classification"))
    # Adding learning rate and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.mean_squared_error,
                metrics=['accuracy'])

    print("-------------------------------------------------")
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
    model_save.saveModel(model,'classification_model')
    predictModel(model,TrainingDataX, TrainingDataY)
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
    model_results.showSummaryClassification(validateDataY,results)
    return results

def predictFullModel(model, data):
    print("\n\n---  Predict Results Classification:          ---")
    dataLabel = tfm.transformDataLabelClassification(data.copy())
    kddCupX = dataLabel
    kddCupX = dataLabel[["protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","lroot_shell","count","diff_srv_rate","dst_host_same_src_port_rate"]]
    kddCupX = kddCupX.to_numpy()
    normalizeDataX = tfm.normalizeColumn(kddCupX)
    normalizeDataX = normalizeDataX.reshape((normalizeDataX.shape[0],normalizeDataX.shape[1],-1))
    results = model.predict(normalizeDataX)
    predictedColumn = np.argmax(results, axis=1)
    data['group_classification'] = predictedColumn
    showResultFullModel(data)
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

def showResultFullModel(data):
    timestr = time.strftime("%Y%m%d-%H%M")
    print("---  Find all the result in the file: results/fullModel"+timestr+".csv      ---")
    array = transfromResult()
    rows = list(set(data["group_classification"].tolist()))
    for row in rows:
        try:
            data["group_classification"] = data["group_classification"].replace([row],array[row])
        except KeyError:
            data["group_classification"] = data["group_classification"].replace([row],'normal')

    # print(kddCup["label"])
    data.to_csv('results/fullModel.csv')

def transfromResult():
    return {
    0: 'normal', 
    1: 'dos',
    2: 'u2r',
    3: 'r2l',
    4: 'probing',
    5: 'smurf',
    6: 'neptune'
    }