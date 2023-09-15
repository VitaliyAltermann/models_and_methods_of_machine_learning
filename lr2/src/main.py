import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=15, input_shape=[1],activation='sigmoid'),
                             keras.layers.Dense(units=1, input_shape=[2],activation='relu')])
model.compile(optimizer='sgd', loss='mean_squared_error')


i=1
xs=np.empty(shape=[1])
ys=np.empty(shape=[1])

#формирование массива
while i<=314:
    xs=np.append(xs,i);
    ys=np.append(ys,np.sin(i/100))
    i=i+1


class GetWeights(tf.keras.callbacks.Callback):
    # Keras callback which collects values of weights and biases at each epoch
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}

    def on_epoch_end(self, epoch, logs=None):
        # this function runs at the end of each epoch
        # loop over each layer and get weights and biases
        for layer_i in range(len(self.model.layers)):
            #w = self.model.layers[layer_i].get_weights()[0][0]
            #w1 = self.model.layers[layer_i].get_weights()[0][1]
            w = self.model.layers[layer_i].get_weights()
            #b = self.model.layers[layer_i].get_weights()[1]
            #print('Layer %s has weights of shape %s and %s and biases of shape %s' %(layer_i, np.shape(w), np.shape(w1), np.shape(b)))
            #print('Layer %s has weights of shape %s and %s and biases of shape %s' % (layer_i))
            print(' %s .  and  %s' %(layer_i, w))
            # save all weights and biases inside a dictionary
            if epoch == 0:
                # create array to hold weights and biases
                self.weight_dict['w_'+str(layer_i+1)] = w
                #self.weight_dict['w1_' + str(layer_i + 1)] = w1
                #self.weight_dict['b_'+str(layer_i+1)] = b
            else:
                # append new weights to previously-created weights array
                for neur_i in range(len(w)):
                    self.weight_dict['w_' + str(layer_i + 1)][neur_i] = np.dstack((self.weight_dict['w_' + str(layer_i + 1)][neur_i], w[neur_i]))

                #self.weight_dict['w1_' + str(layer_i + 1)] = np.dstack((self.weight_dict['w1_' + str(layer_i + 1)], w1))
                # append new weights to previously-created weights array
                #self.weight_dict['b_'+str(layer_i+1)] = np.dstack((self.weight_dict['b_'+str(layer_i+1)], b))


gw = GetWeights()
model.fit(xs, ys, epochs=200, callbacks=[gw])

y=model.predict([1])
print(y)