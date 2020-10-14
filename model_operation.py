# Class to transformate data
import controller.transformation as trans
import controller.normalization as norm
import numpy as np

def transformData(data):
    print("Transform and Normalize Data")
    # Transform text to number
    print("Transform text to number")
    data = trans.transformData(data)
    # Split and remove columns
    print("Remove unused columns")
    kddCupY = data["label"]
    kddCupX = data
    kddCupX.pop("label")
    kddCupX.pop("service")
    # Normalize data
    print("Normalize Colum Data")
    normalizeTrainingDataX = norm.normalizeColumn(kddCupX)
    arrayTrainingDataY = np.array(kddCupY)
    # Reshape
    print("Reshaping Data")
    normalizeTrainingDataX = normalizeTrainingDataX.reshape((normalizeTrainingDataX.shape[0],normalizeTrainingDataX.shape[1],-1))
    return normalizeTrainingDataX, arrayTrainingDataY

def trainModel(epochs,  model, trainingData):
    normalizeTrainingDataX, arrayTrainingDataY = transformData(trainingData)
    # Train Model
    print("\n\nTraining model with new data")
    model.fit(normalizeTrainingDataX,arrayTrainingDataY,epochs=epochs,batch_size=25,shuffle=True)
    result = model.predict(normalizeTrainingDataX)
    print("\n\nResults:")
    print(result)
    return model

def validateModel(model, validateData):
    normalizeValidateDataX, arrayTrainingDataY = transformData(validateData)
    print("\n\nValidate Model with new data")
    result = model.predict(normalizeValidateDataX)
    print("\n\nResults:")
    print(result)
    return model

def confirmValidate():
    confirm = input("\n\n[v]Validate Model or [e]Exit: ")
    if confirm != 'v' and confirm != 'e':
        print("\n\nInvalid Option. Please Enter a Valid Option.")
        confirmValidate() 
    print (confirm)
    return confirm