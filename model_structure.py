
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def defineModel():
    print("\n\n---  Creating RNN Model                       ---")
    # Define new Model for rnn
    model = tf.keras.models.Sequential()
    # Adding model first layer with 41 features (42 - label feature)
    model.add(tf.keras.layers.SimpleRNN(41))
    # Adding Hidden Layers
    # Adding Activation sigmoid on Hidden Layers
    model.add(tf.keras.layers.Dense(units=80,activation='sigmoid',name="dense_1"))
    model.add(tf.keras.layers.Dense(units=160,activation='sigmoid',name="dense_2"))
    model.add(tf.keras.layers.Dense(units=240,activation='sigmoid',name="dense_3"))
    # Adding output layer (normal(0) - anomaly(1))
    model.add(tf.keras.layers.Dense(units=2,activation='softmax',name="predictions"))
    # Adding learning rate and metrics
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                loss=tf.keras.losses.mean_squared_error,
                metrics=['accuracy'])

    print("-------------------------------------------------")
    return model


def defineModelClassification():
    print("\n\n---  Creating LSTM Model                       ---")
    # Define new Model for rnn
    model = tf.keras.models.Sequential()
    # Adding model first layer with 41 features (42 - label feature)
    model.add(tf.keras.layers.LSTM(41))
    # Adding Hidden Layers
    # Adding Activation sigmoid on Hidden Layers
    model.add(tf.keras.layers.Dense(units=80,activation='relu',name="dense_1"))
    model.add(tf.keras.layers.Dense(units=240,activation='relu',name="dense_2"))
    model.add(tf.keras.layers.Dense(units=320,activation='relu',name="dense_3"))
    # Adding output layer (normal(0) - anomaly(1))
    model.add(tf.keras.layers.Dense(units=7,activation='softmax',name="predictions"))
    # Adding learning rate and metrics
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                loss=tf.keras.losses.mean_squared_error,
                metrics=['accuracy'])

    print("-------------------------------------------------")
    return model
