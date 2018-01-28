# coding: utf-8

import sklearn.datasets
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

np.random.seed(0)

# ### step 1 - Build data set
# Make data set
X, y = sklearn.datasets.make_moons(2000, noise=0.20)

print(X[0])
print(y[0])

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
# plt.show()
# plt.close()


# ### step 2 - Build linear model
#
# 먼저 plot_decision_boundary 라는 함수를 정의함.
# 이 함수에 대해서는 이해하지 X
# Classification 모델을 넣으주면 모델이 데이터를 어떻게 나누고 있는 지 시각화해주는 함수임.


def plot_decision_boundary(pred_func):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)



# 이제 keras.utils.np_utils 에서 to_categorical 이라는 함수를 불러온다.
# 이 함수가 어떤 역할을 하는 지는 다음 출력값을 보시면 쉽게 이해하실 수 있음.

y_binary = to_categorical(y)
print("y : ", y)
print("y_binary : ", y_binary)

# 이제 모델을 만들었으니 이 모델을 train 시키기 위한 loss와  optimizer을 설정함.
# 이러한 부분을 쉽게 설정해 주는 것이 바로 model.compile이라는 함수임.

model = Sequential()
model.add(Dense(10, input_dim=2, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(units=2, activation="softmax"))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01, beta_1=0.5),
               metrics=['accuracy'])

model.fit(X, y_binary, epochs=10, batch_size=50)

# Predict and plot
plot_decision_boundary(lambda x: model.predict_classes(x, batch_size=2000))
plt.title("Decision Boundary")
plt.show()
