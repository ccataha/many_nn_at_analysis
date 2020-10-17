
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def defineModel():
    print("---  Creating RNN Model          ---")
    # Define new Model for rnn
    model = tf.keras.models.Sequential()
    # Adding model first layer with 41 features (42 - label feature)
    model.add(tf.keras.layers.SimpleRNN(41))
    # Adding Hidden Layers
    # Adding Activation sigmoid on Hidden Layers
    model.add(tf.keras.layers.Dense(units=80,activation='sigmoid'))
    model.add(tf.keras.layers.Dense(units=160,activation='sigmoid'))
    model.add(tf.keras.layers.Dense(units=240,activation='sigmoid'))
    # Adding output layer (normal(0) - anomaly(1))
    model.add(tf.keras.layers.Dense(units=2,activation='softmax'))
    # Adding learning rate and metrics
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                loss=tf.keras.losses.mean_squared_error,
                metrics=['accuracy'])
    print("---  RNN Model Summary          ---")
    print(model.summary())
    return model
