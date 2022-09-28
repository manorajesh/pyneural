from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import Input
import numpy as np
from math import sqrt

def is_prime(n):
  for i in range(2,int(sqrt(n))+1): # O(sqrt(n))
    if (n%i) == 0:
      return False
  return True

inputs = np.array([])
targets = np.array([])
for i in range(100000):
    rand_int = np.random.randint(1000000)
    inputs = np.append(inputs, rand_int)
    targets = np.append(targets, is_prime(rand_int))

model = Sequential([Input(shape=(1,)), Dense(100), Activation('sigmoid')])

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.summary()
model.fit(inputs, targets, validation_split=0.1, epochs=2, use_multiprocessing=True)

test = np.array([5])
print(model.predict(test))
print(model.evaluate(test))