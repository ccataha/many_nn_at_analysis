
import os
import os.path
from os import path

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers

def saveModel(model, fileName):

    print("\n\n---  Saving Model:                            ---")

    pathFile = 'models/' + fileName + '.h5'
    if path.exists(pathFile):
        os.remove(pathFile)

    model.save(pathFile)

def loadModel(fileName):
    print("\n\n---  Loading Model "+ fileName +" :                  ---")
    pathFile = 'models/' + fileName + '.h5'
    model = None
    if path.exists(pathFile):
        model = keras.models.load_model(pathFile)

    return model
    