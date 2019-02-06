'''

Author :  Fawad Ahmed
Simple custom model for hand written digit recognition.

'''




import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist = tf.keras.datasets.mnist

# Importig data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalizing data
x_train = tf.keras.utils.normalize(x_train)
x_test = tf.keras.utils.normalize(x_test)
# Input shape for flatten layer
input_dem = x_train[0].shape
# model Initialization

model = tf.keras.models.Sequential()
# model architecture....
model.add(tf.keras.layers.Flatten(input_shape=input_dem))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model parameters for training of the model
"""
A neuro network doesnt attempt to optimize accuracy, 
it tries to minimize loss and,
the loss function we chose can impact alot on our neuronet work.    
"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# model training function
model.fit(x_train, y_train, epochs=3)

"""
keep in mind that neru net work are great at fitting data, 
but the questions will our model overfit?.
Our focus should be to make a model so that it can generalize on given Data.
"""
# next thing we do is calculate validation loss and validation accuracy.
val_loss, val_acc = model.evaluate(x_test, y_test)

"""
We should expect our loss to be relativly higher on out of sample data,
Note: we should expect our out of sample accuracy to me slightly lower and our loss to be 
      slightly higher.
      What we dont want to see is either to close or too much of a delta,
      if there is hugh delta chances are you have already overfit your model.
"""
print("Loss : ", val_loss, "Accuracy : ", val_acc)

# Saving a model
model.save("minist_custom.model")
