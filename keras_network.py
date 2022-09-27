from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

inputs = np.array([[2, 3], [3, 10], [6, 8], [100, 82]])
target = np.array([[2.5, 6.5, 7, 91]])

model = Sequential()
model.add(Dense(2, input_shape=(2,)))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy']) 

model.summary()

test = np.array([[2, 3]])
model.predict(test)