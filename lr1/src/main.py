import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[2])])
model.compile(optimizer='sgd', loss='mean_squared_error')



xs = np.array([[-1.0,-1.0], [0.0,0.0], [1.0,1.0], [2.0,2.0], [3.0,3.0], [4.0,4.0]], dtype=float)
ys = np.array([-1.0, 4.0, 9.0, 14.0, 19.0, 24.0], dtype=float)

class GetWeights(tf.keras.callbacks.Callback):
    # Keras callback which collects values of weights and biases at each epoch
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}

    def on_epoch_end(self, epoch, logs=None):
        # this function runs at the end of each epoch
        # loop over each layer and get weights and biases
        for layer_i in range(len(self.model.layers)):
            w = self.model.layers[layer_i].get_weights()[0][0]
            w1 = self.model.layers[layer_i].get_weights()[0][1]
            b = self.model.layers[layer_i].get_weights()[1]
            print('Layer %s has weights of shape %s and %s and biases of shape %s' %(layer_i, np.shape(w), np.shape(w1), np.shape(b)))
            print(' %s . %s,%s and  %s' %(layer_i, w, w1, b))
            # save all weights and biases inside a dictionary
            if epoch == 0:
                # create array to hold weights and biases
                self.weight_dict['w_'+str(layer_i+1)] = w
                self.weight_dict['w1_' + str(layer_i + 1)] = w1
                self.weight_dict['b_'+str(layer_i+1)] = b
            else:
                # append new weights to previously-created weights array
                self.weight_dict['w_'+str(layer_i+1)] = np.dstack((self.weight_dict['w_'+str(layer_i+1)], w))
                self.weight_dict['w1_' + str(layer_i + 1)] = np.dstack((self.weight_dict['w1_' + str(layer_i + 1)], w1))
                # append new weights to previously-created weights array
                self.weight_dict['b_'+str(layer_i+1)] = np.dstack((self.weight_dict['b_'+str(layer_i+1)], b))


gw = GetWeights()
model.fit(xs, ys, epochs=500, callbacks=[gw])