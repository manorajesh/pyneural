from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

inputs = np.array([])
targets = np.array([])

# Create data set to predict average of 2 numbers
for i in range(100000):
    rand_pair = np.random.randint(100, size=2)
    inputs = np.append(inputs, rand_pair)
    inputs = np.reshape(inputs, (-1, 2))
    targets = np.append(targets, np.average(inputs[i]))

model = Sequential()
model.add(Dense(2, input_shape=(2,)))
for i in range(10):
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

#model.add(Activation('sigmoid'))
model.add(Dense(1))
#model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.fit(inputs, targets, validation_split=0.1, epochs=2, use_multiprocessing=True)

#model.summary()

test = np.array([[4, 8]])
print(model.predict(test))
print(model.evaluate(test))