# Library to read excel data
import pandas as pd 
import numpy as np
# Operations
import model_definition as model_df
import model_operation as model_op



print("Welcome!!!")
print("Starting program....")
print("...1")
print("...2")
print("...3")
print("Let's the fun begin....")

# Read csv data
print("\n\nReading csv data")
dirpath = "data/"
filename = "kddcup99_csv.csv"
kddCup = pd.read_csv(dirpath+filename)

# Suffle data
print("Suffle data")
kddCup = kddCup.sample(frac=1)

# Split data in 60% 20% 20%
print("Split data in train, validate and test")
trainingData, validateData, testData = np.split(kddCup.sample(frac=1), [int(.6*len(kddCup)), int(.8*len(kddCup))])

print("trainingData 60% total datos = " + str(len(trainingData)))
print("validateData 20% total datos = " + str(len(validateData)))
print("testData 20% total datos = " + str(len(testData)))

# Suffle Split data
print("\n\nSuffle Split data")
trainingData = trainingData.sample(frac=1)
validateData = validateData.sample(frac=1)
testData = testData.sample(frac=1)


# Define Model Structure
print("\n\nDefining Model")
model = model_df.defineModel()

# Train Model
print("\n\nStart Training Model")
model = model_op.trainModel(3, model, trainingData)

option = model_op.confirmValidate()

if option == "v":
    # Validate Model
    model = model_op.validateModel(model, validateData)
else:
    print("\n\nEnd RNN")



