# программа расчета нейросети, реализующей функцию XOR
import numpy as np

# Обучающая выборка
X = np.array([ [1,0,0],[1,0,1],[1,1,0],[1,1,1] ])
y = np.array([[0,1,1,0]]).T

print(X)
print(y)

# логистическая функция активации
def sigmoidfun(z):
    return 1/(1+np.exp(-z))

def sigmderiv(y):
    return y*(1-y)

# Задаем веса для классификатора, реализующего функцию XOR
syn0 = np.array([ [-5,-10,10],[-5,10,-10] ])
syn1 = np.array([ [-5,10,10] ])

# функция расчета нейросети, реализующей операцию XOR
def xor_nnet(X, syn0, syn1):
    a1 = X.T
    a2 = sigmoidfun(np.dot(syn0, a1))
    ones = np.ones(len(X))
    a20 = np.vstack((ones,a2))
    a3 = sigmoidfun(np.dot(syn1, a20))
    return a1,a20,a3

a1,a20,a3 = xor_nnet(X, syn0, syn1)

print(a3.T - y)
#print(y)

# ------- попробуем обучить нейросеть ---------

# случайно инициализируем веса, в среднем - 0
np.random.seed(3)
syn0 = 2*np.random.random((2,3)) - 1
syn1 = 2*np.random.random((1,3)) - 1

a1,a20,a3 = xor_nnet(X, syn0, syn1)
print(y.T)
print(a3)
#print(a3.T - y)
lrate = 0.1
batch_size = 10

for j in range(120000):
    # проходим вперёд по слоям 0, 1 и 2
    nums = np.random.choice(range(len(X)), batch_size, replace=True)
    Xb = X[nums]
    yb = y[nums]
    Xb = X
    yb = y
    a1, a20, a3 = xor_nnet(Xb, syn0, syn1)

    # как сильно мы ошиблись относительно нужной величины?
    a3_error = yb.T - a3

    # распространяем ошибку на слой назад
    z3_delta = a3_error * sigmderiv(a3)

    # как сильно ошибки в z3 влияют на ошибки в a2?
    a20_error = np.dot(syn1.T, z3_delta)
    a2_error = a20_error[1:]
    a2 = a20[1:]

    # распространяем ошибку на вход слоя
    z2_delta = a2_error * sigmderiv(a2)

    # рассчитаем градиент функционала ошибки по каждому весу
    Dsyn1 = np.dot(a20, z3_delta.T).T
    Dsyn0 = np.dot(a1, z2_delta.T).T

    syn1 += lrate * Dsyn1
    syn0 += lrate * Dsyn0

    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(a3_error))))

a1,a2,a3 = xor_nnet(X, syn0, syn1)
print(y.T)
print(a3)
print(a3.T - y)

print(syn0)
print(syn1)