import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#GET THE MOTHER FUCKING GFLES
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Save as NumPy .npy files
#np.save("x_train.npy", x_train)
#np.save("y_train.npy", y_train)
#np.save("x_test.npy", x_test)
#np.save("y_test.npy", y_test)


# Load file
x_train = np.load("C:/Users/yugaa/Desktop/MINST/src/x_train.npy")

# Check its shape

plt.imshow(x_train[0], cmap="gray")
plt.title("First image")
plt.show()
