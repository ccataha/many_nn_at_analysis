from sklearn import preprocessing
import numpy as np


def normalizeColumn(data):
   arrayData = np.array(data)
   normalizedData = preprocessing.normalize(arrayData)
   return normalizedData