import numpy as np


n_features = 2  # число признаков (размерность входных данных)
n_objects = 300  # число объектов во входных данных
num_epoch = 100  # число эпох обучения / пересчёта

np.random.seed(1) # инициализация генеротора

w_true = np.random.normal(0, 0.1, size=(n_features, ))  # создание истинных весов
w_0 = np.random.uniform(-2, 2, (n_features))            # веса для инициализации - стартовые веса
Xtrain = np.random.uniform(-5, 5, (n_objects, n_features))   # значения Х обучающие
Ytrain = np.dot(Xtrain, w_true) + np.random.normal(0, 1, (n_objects))  # значения Y обучающие

Xtest = np.random.uniform(-5, 5, (n_objects//2, n_features))   # значения Х тестовые
Ytest = np.dot(Xtest, w_true) + np.random.normal(0, 1, (n_objects//2))  # значения Y тестовые

# сохранение массивов

np.save('w_true',w_true)
np.save('w0',w_0)
np.save('Xtrain',Xtrain)
np.save('Ytrain',Ytrain)

np.save('Xtest',Xtest)
np.save('Ytest',Ytest)


