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
- time


### Installation

Clone this repository in your local machine with the following command:

```bash
$ git clone https://github.com/locano/rnn-ids-classifier
```


## Dataset

On this project we use a common dataset [kddcup99](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) alternative [link](https://datahub.io/machine-learning/kddcup99). 


## Usage
```bash
python model_rnn.py
```

## Project Organization
------------
    ├── README.md          <- Readme file to introduces and explains a project.
    │
    ├── tools
    │   ├── data_tool.py   <- Functions to transform data label to numbers.
    │                      <- Functions to normalize transformed data.
    │
    ├── results            <- A output folder for the prediction and classification model.
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
    │   └── results        <- Scripts to create a display matrix results and save/load models.
    │   │   └── model_results.py
    │   │
    │   ├── model_rnn.py   <- Scripts to run project and choose between RNN - Prediction, LSTM - Classification or both models.
    │
    └── .gitignore         <- gitignore file with folders and files not need it on the project.


--------

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
