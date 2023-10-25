import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.datasets import mnist

# загрузка данных с делением на тестовую и обучающёю выборки
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# вывод изображений

# plt.imshow(x_train[1], cmap='gray')
# plt.show()


# параметры модели

numNeur2 = 100  # число нейронов в промежуточном слое
numEpoch = 50  # число эпох обучения
batchSize = 32  # размер пакета
testBatchSize = 32  # размер пакета при тестировании
dropout = 0.15

sizeValid = 5000  # размер валидационной выборки

# создание модели
model = tf.keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),  # входной слой, преобразует визображение в одномерный массив
    keras.layers.Dense(numNeur2, activation='relu'),  # промежуточный слой
    keras.layers.Dropout(dropout),
    keras.layers.Dense(10,
                       activation='softmax')])  # выходной слой выдаёт веротность принадлежности к 10 классам (цифры от 0 до 9)

# компиляция модели

model.compile(optimizer='adam',  # опитимизатор Адам
              loss='categorical_crossentropy',  # функция потерь категориальная
              metrics=['accuracy'])

# обратные вызовы
###########################################################


# обратный вызов при обучении

# class FitCallback(tf.keras.callbacks.Callback):
#    def __init__(self):
#        super(FitCallback, self).__init__()
#        self.callBack_data = {}  # данные извлекаемые из модели
#
#    def on_epoch_end(self, epoch, logs=None):  # после каждой эпохи извлекаются потери
#        if epoch == 0:
#            # создание словарей
#            self.callBack_data['loss'] = logs.get("loss")  # из логов извлекаются потери
#            self.callBack_data['val_loss'] = logs.get("val_loss")
#        else:
#            # дополнение словарей
#            self.callBack_data['loss'] = np.dstack((self.callBack_data['loss'], logs.get("loss")))
#            self.callBack_data['val_loss'] = np.dstack((self.callBack_data['val_loss'], logs.get("val_loss")))


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

# fitCall=FitCallback()
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
his = model.fit(x_train, y_train, epochs=numEpoch, batch_size=batchSize, validation_data=(x_valid, y_valid))

# отрисовка потрь на обучающей и валидационной выборках
plt.title('Потери')

plt.plot(range(0, len(his.history['loss'])), his.history['loss'], label='Обучающая')
plt.plot(range(0, len(his.history['val_loss'])), his.history['val_loss'], label='Валидационная')

plt.legend(loc='best', fontsize=12)
plt.yscale("log")

plt.show()

plt.title('Точность')

plt.plot(range(0, len(his.history['accuracy'])), his.history['accuracy'], label='Обучающая')
plt.plot(range(0, len(his.history['val_accuracy'])), his.history['val_accuracy'], label='Валидационная')

plt.legend(loc='best', fontsize=12)

plt.show()

testRes = model.evaluate(x_test, y_test, batch_size=testBatchSize)

print('Ошибка на тестовых данных ' + str(testRes[0]))
print('Точность на тестовых данных ' + str(testRes[1]))

a = model.predict(x_test[:1])  # предсказание для первого элемента

print()
print('Предсказано ' + str(np.argmax(a)))
print('Верный ответ ' + str(np.argmax(y_test[0])))

print('конец')
