
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def defineModel():
    print("\n\nTensorflow Model:")
    # Define new Model for rnn
    model = tf.keras.models.Sequential()
    # Adding model type with 41 features
    model.add(tf.keras.layers.SimpleRNN(41))
    # Adding Hidden Layers
    model.add(tf.keras.layers.Dense(units=80,activation='sigmoid'))
    model.add(tf.keras.layers.Dense(units=160,activation='sigmoid'))
    model.add(tf.keras.layers.Dense(units=240,activation='sigmoid'))
    # Adding output layer
    model.add(tf.keras.layers.Dense(units=2,activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                loss=tf.keras.losses.mean_squared_error,
                metrics=['accuracy'])

    return model
