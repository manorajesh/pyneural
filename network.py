from curses import erasechar
import numpy as np

input_vector = np.array([1.72, 1.23])
weights_1 = np.array([1.26, 0])
bias = np.array([0.0])
target = 0.0

'''
first_indexes_mult = input_vector[0] * weights_1[0]
second_indexes_mult = input_vector[1] * weights_1[1]
dot_product_1 = first_indexes_mult + second_indexes_mult
'''

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prediction(input_vector, weights, bias):
     layer_1 = np.dot(input_vector, weights) + bias
     layer_2 = sigmoid(layer_1)
     return layer_2

prediction = make_prediction(input_vector, weights_1, bias)

mse = np.square(prediction-target)

derivative = 2 * (prediction - target)

weights_1 = weights_1 - derivative

prediction = make_prediction(input_vector, weights_1, bias)
error = np.square(prediction - target)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

derror_dprediction = 2 * (prediction - target)
layer1 = np.dot(input_vector, weights_1) + bias
dprediction_dlayer1 = sigmoid_derivative(layer1)
dlayer1_dbias = 1

derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias

print(f"Prediction: {prediction}; Error: {error}")