# программа расчета нейросети, реализующей функцию XOR
import numpy as np

# Обучающая выборка
X = np.array([ [1,0,0],[1,0,1],[1,1,0],[1,1,1] ])
y = np.array([[0,1,1,0]]).T

print(X)
print(y)

# -------------- Моделирование классификатора XOR -------------
# логистическая функция активации
def sigmoidfun(z):
    return 1 / (1 + np.exp(-z))


def sigmderiv(z):
    return sigmoidfun(z) * (1 - sigmoidfun(z))

def sigmoideriv(y):
    return y * (1 - y)

np.random.seed(1)
s0 = 2 * np.random.random((2, 3)) - 1
s1 = 2 * np.random.random((1, 3)) - 1

def xor_nnet(x, syn0=s0, syn1=s1):
    a1 = x.T
    a2 = sigmoidfun(np.dot(syn0, a1))
    if len(a2.shape) > 1:
        ones = np.ones(shape=(1, a2.shape[1]))
        a20 = np.vstack((ones, a2))
    else:
        a20 = np.array([1, a2[0], a2[1]])
    a3 = sigmoidfun(np.dot(syn1, a20))
    return a1, a20, a3

# -------------- Моделирование классификатора XOR -------------
# Задаем веса для классификатора, реализующего функцию XOR
syn0 = np.array([ [-5,-10,10],[-5,10,-10] ])
syn1 = np.array([ [-5,10,10] ])

ynet = np.array([[0.0, 0.0, 0.0, 0.0]]).T
err = ynet
for i in range(len(X)):
    x = X[i]
    a1,a2,ynet[i] = xor_nnet(x, syn0, syn1)
    err[i] = y[i] - ynet[i]
    print("Ответ классификатора для", x, "=", ynet[i])

# --------------- анализ ошибок классификации -----------------
# СРЕДНЕКВАДРАТИЧЕСКАЯ ОШИБКА (MSE)
print('MSE=', sum(err*err)/len(err))
# print('MSE=', np.mean(err*err))

# кроссэнтропия распределений (log-loss)
def plogp(p):
    if abs(p) < 1e-12:
        return 0
    return p*np.log2(p)

# ------- попробуем обучить нейросеть простым град-м спуском ---------
# Попробуем обучить нейросеть XOR с помощью оптимизации
# случайно инициализируем веса, в среднем - 0
np.random.seed(3)
syn0 = 2*np.random.random((2,3)) - 1
syn1 = 2*np.random.random((1,3)) - 1

a1,a2,ynet = xor_nnet(X, syn0, syn1)
print(ynet)
# learning coef
alpha = 0.01

# Попробуем обучить нейросеть XOR
for j in range(600000):
    # проходим вперёд по слоям 0, 1 и 2
    a10,a20,a3 = xor_nnet(X, syn0, syn1)

    # как сильно мы ошиблись относительно нужной величины?
    a3_error = y - a3.T

    # рапространим ошибку на выходе 3-го слоя на вход слоя
    z3_delta = a3_error * sigmoideriv(a3.T)

    # распространим ошибки z3 на ошибки выходов a2 (входы для a3)
    a20_error = np.dot(syn1.T, z3_delta.T)
    # уберем 0-ю строку, содержащую добавленный +1
    a2 = a20[1:]
    a2_error = a20_error[1:]

    # рапространим ошибку на выходе 2-го слоя на вход слоя
    z2_delta = a2_error * sigmoideriv(a2)

    # рассчитаем дельта весов каждого слоя,
    # используя ошибки на входе (l+1) слоя и выходы l-го слоя
    syn1_delta = np.dot(z3_delta.T, a20.T)
    syn0_delta = np.dot(z2_delta, a10.T)

    syn1 += alpha* syn1_delta
    syn0 += alpha* syn0_delta

    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(a3_error))))

print(syn1)
print(syn0)
a1,a2,ynet = xor_nnet(X, syn0, syn1)
print(ynet)
"""
Error:0.501780663237
Error:0.499985281905
Error:0.499510817604
Error:0.495886793439
Error:0.462283439271
Error:0.388309485676
Error:0.344607287938
Error:0.21591328267
Error:0.124516190796
Error:0.0912743545313
Error:0.0743180365023
...
Error:0.0161680950144
Error:0.0160026083371
syn0=
[[-7.00981282  4.58522782  4.55025869]
 [-2.98183124  6.76632391  6.60203154]]
syn1=
[[ -4.48638541 -10.51846703   9.74164621]]

"""

# ------- попробуем обучить нейросеть стохастическим градиентом ---------
# случайно инициализируем веса, в среднем - 0
np.random.seed(3)
syn0 = 2*np.random.random((2,3)) - 1
syn1 = 2*np.random.random((1,3)) - 1

# learning coef
alpha = 0.01
# pack size
pack_size = 3

# Попробуем обучить нейросеть XOR
for j in range(600000):
    # проходим вперёд по слоям 0, 1 и 2
    a10,a20,a3 = xor_nnet(X, syn0, syn1)
    # выбираем случайно pack_size объектов из обучающей выборки
    nobjs = np.random.choice(list(range(len(X))), pack_size, replace=False)
    # фильтруем нужные объекты и ответы
    a10 = a10[:, nobjs]
    a20 = a20[:, nobjs]
    a3 = a3[:, nobjs]
    yp = y[nobjs]

    # как сильно мы ошиблись относительно нужной величины?
    a3_error = yp - a3.T

    # рапространим ошибку на выходе 3-го слоя на вход слоя
    z3_delta = a3_error * sigmoideriv(a3.T)

    # распространим ошибки z3 на ошибки выходов a2 (входы для a3)
    a20_error = np.dot(syn1.T, z3_delta.T)
    # уберем 0-ю строку, содержащую добавленный +1
    a2 = a20[1:]
    a2_error = a20_error[1:]

    # рапространим ошибку на выходе 2-го слоя на вход слоя
    z2_delta = a2_error * sigmoideriv(a2)

    # рассчитаем дельта весов каждого слоя,
    # используя ошибки на входе (l+1) слоя и выходы l-го слоя
    syn1_delta = np.dot(z3_delta.T, a20.T)
    syn0_delta = np.dot(z2_delta, a10.T)

    syn1 += alpha* syn1_delta
    syn0 += alpha* syn0_delta

    if (j % 50000) == 0:
        print("Error:" + str(np.mean(np.abs(a3_error))))

print(syn1)
print(syn0)
a1,a2,ynet = xor_nnet(X, syn0, syn1)
print(a1)
print(a2)
print(ynet)


"""
Результат получился только после снижения альфа
alpha = 0.01
pack_size = 3

Error:0.608719767912
Error:0.481788312008
Error:0.122990617585
Error:0.0526038141682
Error:0.0384722962964
Error:0.0308505911275
Error:0.0278507120516
Error:0.0244147135194
Error:0.0225262690888
Error:0.0205620923888
Error:0.0194889004117
syn0=
[[-6.81597651  4.46157227  4.42166219]
 [-2.92654715  6.69726717  6.5037521 ]]
syn1=
[[ -4.31594635 -10.19845644   9.41363496]]

"""

# ---------- реализация нейросети XOR с помощью KERAS
# import the necessary packages
from keras.optimizers import SGD, Adam, Nadam
from keras.utils import np_utils

# imports used to build the deep learning model
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import Input, Dense
from keras.layers import BatchNormalization, Dropout, Lambda
from keras.models import Model

# from keras. import binary_crossentropy
from keras import backend as K

import matplotlib.pyplot as plt

def graph_training_history(history):
    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

def build_XORnnet(nx = 2):
    # Input
    a1 = Input(shape=(nx,))
    a2 = Dense(2, activation='sigmoid')(a1)
    a3 = Dense(1, activation='sigmoid')(a2)

    classificator = Model(a1, a3, name="XORnnet")
    # Return the constructed network architecture
    return classificator

model = build_XORnnet(nx = 2)
# Посмотрим на число параметров
model.summary()

# Зададим основные параметры для обучения
batch_size = 4
Xb = X[:,1:]
yb = y

# Build and Compile the model
print("[INFO] Building and compiling the DNet model...")
#opt = SGD(lr=0.0002, momentum=0.0001, decay=0.00001, nesterov=True)
opt = Adam(lr=0.01)
#opt = Nadam(lr=0.0001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Check the argument whether to train the model
print("[INFO] Training the model...")
history = model.fit(Xb, yb,
                    batch_size= batch_size,
                    epochs=50,
                    shuffle= True,
                    #validation_split= 0.3,
                    validation_data=(Xb, yb),
                    verbose=1)

# Visualize the training history
graph_training_history(history)

# Training of the model is now complete
#model.summary()

# Use the test data to evaluate the model
# check it on ctrldata - проверяем точность модели на контрольной выборке
#ctrlLabels_softmax = np_utils.to_categorical(ctrlLabels, n_classes)
print("[INFO] Evaluating the model...")
(loss, accuracy) = model.evaluate(ctrlData, ctrlLabels_softmax, verbose=1, batch_size= batch_size)
print("[INFO] CONTROL accuracy: {:.2f}%".format(accuracy * 100))

