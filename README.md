# RNN-IDS-Classifier


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About The Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Environment](#environment)
  * [Dependencies](#dependencies)
  * [Installation](#installation)
* [Dataset](#dataset)
* [Usage](#usage)
* [Contributing](#contributing)
* [Author](#author)
* [References](#references)


## About The Project

On this repository you will find an example of how to build a recurrent neural network using Python, Tensorflow and Keras. 

This is my first contribution to a machine learning project, hope this project can be helpful to people interested in this topic.


## Getting Started

To get a local copy up and running follow these simple steps:


#### Environment

- Python 3.7
- Conda


#### Dependencies

- tensorFlow
- keras
- numpy
- sklearn
- os
- pandas


## Project Organization
------------
    ├── README.md          <- Readme file to introduces and explains a project.
    │
    ├── tools
    │   ├── data_tool.py   <- Functions to transform data label to numbers.
    │                      <- Functions to normalize transformed data.
    │
    ├── result            <- A output folder for the prediction and classification model.
    │   ├── modelResult.csv                <- Model result output.
    │
    ├── resources          <- Datasets for train and test our model.
    │   ├── kddcup99_csv_balance.csv       <- Data set used for train classification model.
    │   ├── kddcup99_csv.csv               <- Data set used for train prediction model.
    │   ├── test_data.csv                  <- Data set used for test models.
    │
    ├── models             <- Trained models, prediction and classification.
    │   ├── classification_model.h5        <- Classification model.
    │   ├── prediction_model.h5            <- Prediction model.
    │
    ├── references.txt     <- Links Reference.
    │
    ├── src                <- Source code.
    │   │
    │   ├── data           <- Scripts to read and generate data
    │   │   └── model_data.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and used for predictions
    │   │   ├── model_classification.py
    │   │   └── model_prediction.py
    │   │
    │   └── result        <- Scripts to create a display matrix results and save/load models.
    │   │   └── model_results.py
    │   │
    │   ├── model_rnn.py   <- Scripts to run project and choose between RNN - Prediction, LSTM - Classification or both models.
    │
    └── .gitignore         <- gitignore file with folders and files not need it on the project.


--------


### Installation

Clone this repository in your local machine with the following command:

```bash
$ git clone https://github.com/locano/rnn-ids-classifier
$ conda create -n rnn-ids python=3.7
$ conda activate rnn-ids
$ pip install tensorflow
$ pip install pandas
$ pip install sklearn
```


## Dataset

On this project we use a common dataset [kddcup99](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) alternative [link](https://datahub.io/machine-learning/kddcup99). 

If you want to change the train data or add your *validation test data* please add it under:

------------
    ├── resources          <- Datasets for train and test our model.
    │   ├── kddcup99_csv_balance.csv       <- Data set used for train classification model.
    │   ├── kddcup99_csv.csv               <- Data set used for train prediction model.
    │   ├── test_data.csv                  <- Data set used for test models.
-----

This route is used on the project to make the train and finally implement the prediction model.


## Usage
```
$ python model_rnn.py
$ Starting Neural Network
$ What kind of NN you want to implement?
$ --> RNN - Train Prediction [1]
$ --> LSTM - Train Classification [2]
$ --> Predict, Validate RNN and LSTM Model [3]
```

**RNN - Prediction** 
```
$ Do you want to Train[1], Exit[2]?
```

Once you select this option, our model will start to read and prepare the data coming from *kddcup99_csv*.
In this option, the data will split into three blocks, training, validation, and test. Now with the training block, we transform and normalize the features included in the data_sources.
For this model, we select all the features (41).

***Model definition***

Then the model is defined as a Sequential with the first layer as a SimpleRNN with forty-one input features. 
After a few tests, we define our model with five Hidden Layer.
- Three Dense Layers (82, 164, 248 units)
- Two BatchNormalization Layers
Then defined or output Layer with two units with softmax activation..
And selected sigmoid as activation and, we use Adam as an optimizer.

Finally, we train our data with five epochs and save the predicted model *prediction_model.h5*.
 

**LSTM - Classification** 
```
$ Do you want to Train[1], Exit[2]?
```
Once you select this option, our model will start to read and prepare the data coming from *kddcup99_balance_csv*.

In this option, the data will split into three blocks, training, validation, and test. Now with the training block, we transform and normalize the features included in the data_sources.

For this model, we select eleven features determined with Weka.
- protocol_type
- service
- flag
- src_bytes
- dst_bytes
- land
- wrong_fragment
- lroot_shell
- count
- diff_srv_rate
- dst_host_same_src_port_rate

***Model definition***

Then the model is defined as a Sequential with the first layer as an LSTM with eleven input features. 

After a few tests, we define our model with five Hidden Layer.
- Five Dense Layers (22, 44, 88, 176, 264 units)
Then defined or output Layer with seven units with softmax activation.
And selected softsign as activation and, we use Adam as an optimizer.

Finally, we train our data with three epochs and save the predicted model *classification_model.h5*.


**Predict, Validate RNN and LSTM Model**    
```
$ Enter File Name: ?
```
Once you select this option, you will prop to input a value. *test_data.csv* or the data you want to predict.
Now we load our two previous trained models, prediction, and classification. *prediction_model.h5* *classification_model.h5*
Our model will start to read and prepare the data.

Then with the *prediction_model* loaded, we predict our PredictedColum this column will determine the attacks predicted as normal(0) or anomaly(1).

Then we use the PredictedColumn to remove all the attacks defined as 'normal' and start the prediction with our *classification_model* loaded.

Finally, the predicted result gets saved in our 'results' folder with the name modelResult.csv


## Contributing

Please feel free to contribute or comment on anything about coding style or algorithm suggestions on this repository.
Any contributions you make are **appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/ContributionFeature`)
3. Commit your Changes (`git commit -m 'Add some ContributionFeature'`)
4. Push to the Branch (`git push origin feature/ContributionFeature`)
5. Open a Pull Request


## Authors
- Ludwing Cano - [locano](https://github.com/locano)


## Reference
- [Tensorflow](https://www.tensorflow.org/learn)
- [Keras](https://keras.io/guides/)
- [Weka](https://www.cs.waikato.ac.nz/ml/weka/)
- [How to Train a model](https://keras.io/guides/training_with_built_in_methods/)
- [Optimal Number of epochs](https://www.geeksforgeeks.org/choose-optimal-number-of-epochs-to-train-a-neural-network-in-keras/)
- [Categorical Function](https://keras.io/api/utils/python_utils/#to_categorical-function)
- [Keras Batch Normalization](https://keras.io/api/layers/normalization_layers/batch_normalization/)
- [Keras Activatios](https://keras.io/api/layers/activations/)
- [Pandas DataFrame Numpy](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_numpy.html)
