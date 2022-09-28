from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import Input
import numpy as np
from math import sqrt

## Spinner
import sys
import time
import threading

class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1: 
            for cursor in '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏': yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False

def is_prime(n):
  for i in range(2,int(sqrt(n))+1): # O(sqrt(n))
    if (n%i) == 0:
      return False
  return True

with Spinner():
    inputs = np.array([])
    targets = np.array([])
    for _ in range(100000):
        rand_int = np.random.randint(1000000)
        inputs = np.append(inputs, rand_int)
        targets = np.append(targets, is_prime(rand_int))
print("\r\n")

model = Sequential()
model.add(Dense(4, input_shape=(1,), activation='sigmoid', kernel_regularizer='l2'))
for _ in range(10):
    model.add(Dense(128, activation='sigmoid', kernel_regularizer='l2'))
model.add(Dense(1, activation='relu', ))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(inputs, targets, validation_split=0.2, epochs=200, use_multiprocessing=True, batch_size=1000)

with Spinner():
    test = np.array([])
    for _ in range(100):
        rand_int = np.random.randint(100)
        test = np.append(inputs, rand_int)
print("\r\n")

predictions = model.predict(test)
for i in range(len(test)):
    print(f"test={test[i]}, prediction={predictions[i]}")

model.evaluate(test, targets)

model.save('model')
print("Model saved")