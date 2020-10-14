
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def defineModel():
    print("\n\nTensorflow Model:")
    # Define new Model for rnn
    model = tf.keras.models.Sequential()
    # Adding model type
    model.add(tf.keras.layers.SimpleRNN(4))
    # Adding Hidden Layers
    model.add(tf.keras.layers.Dense(units=3,activation='sigmoid'))
    model.add(tf.keras.layers.Dense(units=5,activation='sigmoid'))
    model.add(tf.keras.layers.Dense(units=8,activation='sigmoid'))
    # Adding output layer
    model.add(tf.keras.layers.Dense(units=2,activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                loss=tf.keras.losses.mean_squared_error,
                metrics=['accuracy'])
    return model
