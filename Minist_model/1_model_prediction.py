import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist
# Importig data set
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# loading model 
model = tf.keras.models.load_model("minist_custom.model")

# prediction
pred = model.predict([x_test])

print(np.argmax(pred[0]))


plt.imshow(x_test[0])
plt.show()
