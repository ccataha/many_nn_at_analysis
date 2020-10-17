import model_data as data_processing
import model_structure as model_structure
import model_prediction as model_prediction
import model_classification as model_classification

# Program header
print("---  Simple RNN Neural Network                ---")
print("---  Starting Neural Network                  ---")
print("-------------------------------------------------")

# Get Data
trainingData, validateData, testData = data_processing.readData()

# # Create Model Prediction
# model = model_structure.defineModel()
# # Train and Validate Model
# model = model_prediction.trainModel(3, model, trainingData, validateData)

# Create Model Classification
model = model_structure.defineModelClassification()
# Train and Validate Model
model = model_classification.trainModel(3, model, trainingData, validateData)
