# Library to create rnn model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Library to read excel data
import pandas as pd 
# Class to transformate data
import controller.transformation as trans
import controller.normalization as norm
import model_operation as operation
import numpy as np

# Read csv data
dirpath = "data/"
filename = "kddcup99_csv.csv"
kddCup = pd.read_csv(dirpath+filename)

# Suffle data
kddCup = kddCup.sample(frac=1)
# Split data in 60% 20% 20%
trainingData, validateData, testData = np.split(kddCup.sample(frac=1), [int(.6*len(kddCup)), int(.8*len(kddCup))])
# Suffle data
trainingData = trainingData.sample(frac=1)
validateData = validateData.sample(frac=1)
testData = testData.sample(frac=1)

# Transform text to number
trainingData = trans.transformData(trainingData)
# Split and remove columns
kddCupY = trainingData["label"]
kddCupX = trainingData
kddCupX.pop("label")
kddCupX.pop("service")
# Normalize data
normalizeTrainingDataX = norm.normalizeColumn(kddCupX)
arrayTrainingDataY = np.array(kddCupY)
# Reshape
normalizeTrainingDataX = normalizeTrainingDataX.reshape((normalizeTrainingDataX.shape[0],normalizeTrainingDataX.shape[1],-1))

# Define new Model for rnn
model = tf.keras.models.Sequential()
# Adding model type
model.add(tf.keras.layers.SimpleRNN(2))
# Adding Hidden Layers
model.add(tf.keras.layers.Dense(units=3,activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=5,activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=8,activation='sigmoid'))
# Adding output layer
model.add(tf.keras.layers.Dense(units=2,activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=['accuracy'])
epochs = 3
# Train Model
model.fit(normalizeTrainingDataX,arrayTrainingDataY,epochs=epochs,batch_size=25,shuffle=True)
print("\n\nResultados en entrenamiento:")
result = model.predict(normalizeTrainingDataX)
print(result)

# Validate Model
operation.validateModel(model, validateData)



