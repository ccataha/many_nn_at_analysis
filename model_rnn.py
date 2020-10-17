import model_data as data_processing
import model_structure as model_structure
import model_operation as model_operation

# Program header
print("---  Simple RNN Neural Network                ---")
print("---  Starting Neural Network                  ---")
print("-------------------------------------------------")

# Get Data
trainingData, validateData, testData = data_processing.readData()
# Create Model
model = model_structure.defineModel()
# Train and Validate Model
model = model_operation.trainModel(3, model, trainingData, validateData)
# Evaluate Model
model_operation.evaluateModel(model, testData)
