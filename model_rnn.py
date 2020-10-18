import model_data as data_processing
import model_prediction as model_prediction
import model_classification as model_classification

# Program header
print("---  Starting Neural Network                  ---")
print("-------------------------------------------------")

# Get Data
# trainingData, validateData, testData = data_processing.readData("kddcup99_csv.csv")
# # Create Model Prediction
# model = model_prediction.defineModel()
# # Train and Validate Model
# model = model_prediction.trainModel(3, model, trainingData, validateData)


# Get Data
trainingData, validateData, testData = data_processing.readData("kddcup99_csv_balance.csv")

# Create Model Classification
model = model_classification.defineModel()
# Train and Validate Model
model = model_classification.trainModel(3, model, trainingData, validateData)
