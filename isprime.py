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
for _ in range(100000):
    rand_int = np.random.randint(1000000)
    inputs = np.append(inputs, rand_int)
    targets = np.append(targets, is_prime(rand_int))

model = Sequential()
model.add(Dense(6, input_shape=(1,), activation='sigmoid', kernel_regularizer='l2'))
for i in range(4, 0, -1):
    model.add(Dense(6//i, activation='sigmoid', kernel_regularizer='l2'))
model.add(Dense(1, activation='relu', ))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(inputs, targets, validation_split=0.2, epochs=200, use_multiprocessing=True)

test = np.array([])
for _ in range(100):
    rand_int = np.random.randint(100)
    test = np.append(inputs, rand_int)

predictions = model.predict(test)
for i in range(len(test)):
    print(f"test={test[i]}, prediction={predictions[i]}")

model.save('model')