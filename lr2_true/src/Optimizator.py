import math
import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras

nOpt = 1  # Номер оптимизатора

n_features = 2  # число признаков (размерность входных данных)
n_objects = 300  # число объектов во входных данных
num_epoch = 50  # число эпох обучения / пересчёта

batchSize = 32  # размер пакета для обучения

w_true = np.load('w_true.npy')  # реальные веса


# функция инициализации весов
def my_init(shape, dtype=None):
    w01 = np.empty((2, 1))
    w0 = np.load('w0.npy')
    w01[0][0] = w0[0]
    w01[1][0] = w0[1]

    return w01  # вычитывает веса из файла


# создание модели - один нейрон с линейной активацией для невлияния на параметры
model = tf.keras.Sequential(
    [keras.layers.Dense(1, input_shape=(n_features,), activation='linear', kernel_initializer=my_init)])

# добавление параметрического оптимизатора

if nOpt == 1:
    # Adam
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)  # параметр альфа - скорость изменения
elif nOpt == 2:
    # Momentum
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5,
                                  name='Momentum')  # параметр альфа - скорость изменения
elif nOpt == 3:
    # NAG
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=True,
                                  name='NAG')  # параметр альфа - скорость изменения
elif nOpt == 4:
    # RMSprop
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)  # параметр альфа - скорость изменения
else:
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)  # параметр альфа - скорость изменения

# создание модели
model.compile(optimizer=opt, loss='mean_squared_error')


################################################################################
class GetWeights(tf.keras.callbacks.Callback):
    # функция, вызывающаяся при завершении каждой эпохи, позволяет извлекать веса, ошибку и др
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}

    def on_epoch_end(self, epoch, logs=None):
        # в конце эпохи

        # цикл по всем слоям для извлечения весов
        for layer_i in range(len(self.model.layers)):
            w = self.model.layers[layer_i].get_weights()[0]
            b = self.model.layers[layer_i].get_weights()[1]

            # сохранение весов
            if epoch == 0:
                # добавление весов для каждого слоя в словарь
                self.weight_dict['w_' + str(layer_i + 1)] = w
                self.weight_dict['b_' + str(layer_i + 1)] = b
            else:
                # добавление весов для каждого слоя в словарь
                self.weight_dict['w_' + str(layer_i + 1)] = np.dstack((self.weight_dict['w_' + str(layer_i + 1)], w))
                self.weight_dict['b_' + str(layer_i + 1)] = np.dstack((self.weight_dict['b_' + str(layer_i + 1)], b))
        # добавление ошибки в словарь
        if epoch == 0:
            self.weight_dict['l'] = logs.get("loss")
        else:
            self.weight_dict['l'] = np.dstack((self.weight_dict['l'], logs.get("loss")))


gw = GetWeights()


##################################################################################


# Функция отрисовка траетории
def plot_gd(X, y, w_history):
    # compute level set
    A, B = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

    levels = np.empty_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            w_tmp = np.array([A[i, j], B[i, j]])
            levels[i, j] = np.mean(np.power(np.dot(X, w_tmp) - y, 2))

    plt.figure(figsize=(13, 9))
    plt.title(opt.name + ' размер пакета ' + str(batchSize) + ' траектория')
    plt.xlabel(r'$w_1$')
    plt.ylabel(r'$w_2$')
    plt.xlim((-2.1, 2.1))
    plt.ylim((-2.1, 2.1))

    # visualize the level set
    CS = plt.contour(A, B, levels, levels=np.logspace(0, 2, num=10), cmap=plt.cm.rainbow_r)
    CB = plt.colorbar(CS, shrink=0.8, extend='both')

    # visualize trajectory
    w_list = np.array(w_history)
    plt.scatter(w_true[0], w_true[1], c='r', marker='*')
    plt.scatter(w_list[0, :], w_list[1, :])
    plt.plot(w_list[0, :], w_list[1, :])
    plt.show()


##########################################################################################

# Загрузка массивов данных
Xtrain = np.load('Xtrain.npy')
Ytrain = np.load('Ytrain.npy')

Xtest = np.load('Xtest.npy')
Ytest = np.load('Ytest.npy')

timeStart = datetime.datetime.now()  # метска времени перед обучением
# обучение модели
model.fit(Xtrain, Ytrain, epochs=num_epoch, batch_size=batchSize, callbacks=[gw])
timeStop = datetime.datetime.now()  # метска времени после обучением

# сохранение графика потерь в файл
np.save('loss' + opt.name + str(batchSize), gw.weight_dict['l'][0][0])

# отрисовка графика
plot_gd(Xtrain, Ytrain, gw.weight_dict['w_1'][:, 0])

# вычисление ошибки на тестовых данных
er = 0

for i in range(len(Xtest)):
    y = model.predict(np.reshape(Xtest[0], (2, 1)).T)
    er = er + (y - Ytest[i]) * (y - Ytest[i])

print()
print('Время обучения')
print(timeStop - timeStart)
print()

print('Ошибка на обучающих данных')
print(gw.weight_dict['l'][0][0][[num_epoch - 1]])
print()

print('Ошибка на тестовых данных')
print(er)
