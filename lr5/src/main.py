import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.datasets import mnist
from keras import backend as K

# загрузка данных с делением на тестовую и обучающёю выборки
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# вывод изображений

# plt.imshow(x_train[1], cmap='gray')
# plt.show()


# параметры модели

numFilter = 8  # число фильтров в первом слое
sizeFilter = 3  # размер фильтра в первом слое
numFilter2 = 16  # число фильтров во втором слое
sizeFilter2 = 3  # размер фильтра во втором слое
numEpoch = 1  # число эпох обучения
batchSize = 128  # размер пакета
testBatchSize = 32  # размер пакета при тестировании
# dropout=0.15

sizeValid = 5000  # размер валидационной выборки

# создание модели
model = tf.keras.Sequential([
    keras.layers.Conv2D(numFilter,
                        (sizeFilter, sizeFilter),
                        padding='same',
                        activation='relu',
                        input_shape=(28, 28, 1)),  # свертка
    keras.layers.MaxPooling2D((2, 2), strides=2),
    # замена 2 на 2  на максимальное значение из этих ячеек, шаг 2 по обеим осям
    keras.layers.Conv2D(numFilter2,
                        (sizeFilter2, sizeFilter2),
                        padding='same',
                        activation='relu'),  # свёртка
    keras.layers.MaxPooling2D((2, 2), strides=2),
    keras.layers.Flatten(),  # вытягивание в один вектор
    keras.layers.Dense(128, activation='relu'),
    # выходной слой выдаёт веротность принадлежности к 10 классам (цифры от 0 до 9)
    keras.layers.Dense(10, activation='softmax')])

# компиляция модели

model.compile(optimizer='adam',  # опитимизатор Адам
              loss='categorical_crossentropy',  # функция потерь категориальная
              metrics=['accuracy'])

filtr = []  # переменная под два фильтра для ручной свёртки


# обратные вызовы
###########################################################


# обратный вызов при обучении

class FitCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(FitCallback, self).__init__()
        self.callBack_data = {}  # данные извлекаемые из модели

    # в конце каждой эпохи визуализируются веса
    def on_epoch_end(self, epoch, logs=None):
        if (epoch == numEpoch - 1):

            for layer_i in [0, 2]:  # для 0-го и второго слоя (слои свёртки)
                filters, biases = self.model.layers[layer_i].get_weights()  # извлечение весов

                f_min, f_max = filters.min(), filters.max()
                filters = (filters - f_min) / (f_max - f_min)  # перевод в значения от 0 до 1

                shape = filters.shape
                n_filters, ix, iy, cols = shape[3], 0, 0, 8
                _, axs = plt.subplots((shape[3] // cols) + 1, cols)  # разбиение на подобласти рисования
                for i in range(n_filters):
                    f = filters[:, :, :, i]  # получение весов одного фильтра
                    if iy >= cols:
                        ix += 1
                        iy = 0
                    ax = axs[ix, iy]
                    ax.imshow(f[:, :, 0], cmap='gray')  # добавление на отрисовку
                    iy += 1
                plt.show()

                if layer_i == 0:
                    # вытаскивание двух фильтров
                    filtr.append(filters[:, :, :, 1])
                    filtr.append(filters[:, :, :, 2])


# обратный вызов при тесте

# class TestCallback(tf.keras.callbacks.Callback):
#    def __init__(self):
#        super(TestCallback, self).__init__()
#        self.callBack_data = {}
#
#    def on_test_batch_end(self, batch, logs=None):  # после каждого тестового пакета извлекаются данные о потерях
#        if self.callBack_data == {}:
#            self.callBack_data['loss'] = logs.get("loss")
#        else:
#            self.callBack_data['loss'] = np.dstack((self.callBack_data['loss'], logs.get("loss")))


########################################################################

# визуализация фильтров и карт

def mapVisual(filters):
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)  # перевод в значения от 0 до 1

    shape = filters.shape
    n_filters, ix, iy, cols = shape[3], 0, 0, 8
    _, axs = plt.subplots((shape[3] // cols) + 1, cols)  # разбиение на подобласти рисования
    for i in range(n_filters):
        f = filters[:, :, :, i]  # получение весов одного фильтра
        if iy >= cols:
            ix += 1
            iy = 0
        ax = axs[ix, iy]
        ax.imshow(f[0, :, :], cmap='gray')  # добавление на отрисовку
        iy += 1
    plt.show()


###################################################

# функция свёертки с визуализацией

def svertka(img, core):
    image = np.zeros((1, 30, 30))
    for i in range(0, img.shape[1]):
        for j in range(0, img.shape[2]):
            image[0, i + 1, j + 1] = img[0, i, j]

    res = np.zeros((1, img.shape[1], img.shape[2]))

    for i in range(0, img.shape[1], 2):
        for j in range(0, img.shape[2], 2):
            res[0, i, j] = image[0, i, j] * core[0, 0, 0] + image[0, i + 1, j] * core[1, 0, 0] + image[0, i + 2, j] * \
                           core[2, 0, 0] + image[0, i, j + 1] * core[0, 1, 0] + image[0, i + 1, j + 1] * core[1, 1, 0] + \
                           image[0, i + 2, j + 1] * core[2, 1, 0] + image[0, i, j + 2] * core[0, 2, 0] + image[
                               0, i + 1, j + 2] * core[1, 2, 0] + image[0, i + 2, j + 2] * core[2, 2, 0]

    plt.imshow(res[0], cmap='gray')
    plt.show()

    a = 5


############################################################


fitCall = FitCallback()

# testCall=TestCallback()

# преобразование в категориальные признаки (реализует двоичное позиционное унитарное кодирование признака (бит с индексом цифры равен 1, остальные 0))

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# разделение обучающей на обучающую и валидационную выборки

x_valid = x_train[:sizeValid]  # взятие первых sizeValid элементов
y_valid = y_train[:sizeValid]

x_train = x_train[sizeValid:]
y_train = y_train[sizeValid:]

# обучеие модели
his = model.fit(x_train, y_train, epochs=numEpoch, batch_size=batchSize, validation_data=(x_valid, y_valid),
                callbacks=fitCall)

testRes = model.evaluate(x_test, y_test, batch_size=testBatchSize)

print('Ошибка на тестовых данных ' + str(testRes[0]))
print('Точность на тестовых данных ' + str(testRes[1]))

# предсказание

for j in [0, 1]:
    a = model.predict(x_test[j].reshape((1, 28, 28)))  # предсказание для первого элемента

    print()
    print('Предсказано ' + str(np.argmax(a)))
    print('Верный ответ ' + str(np.argmax(y_test[j])))

# создание функции, которая возвращает выход слоя nLayer

# nLayer=1

for i in range(0, 4):
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[i].output])

    # вызов функции с передачей изображения
    layer_output = get_3rd_layer_output(x_test[0:1])[0]

    # визуализация карт признаков
    mapVisual(layer_output)

# самостоятельная свёртка
svertka(x_test[1].reshape((1, 28, 28)), filtr[0])

print('FINISH')
