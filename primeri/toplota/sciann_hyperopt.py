import numpy as np
import matplotlib.pyplot as plt 
import sciann as sn
from numpy import pi
from sciann.utils.math import diff, sign, sin, sqrt, exp
import random
# from tensorflow.keras.callbacks import EarlyStopping
import os
import time
import math
import sys
import base64
import json
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle


from hyperopt import space_eval
from hyperopt import fmin, hp, tpe
from hyperopt import SparkTrials, STATUS_OK, STATUS_FAIL


def calculate_loss(model : sn.SciModel, x_data, t_data):

    predicted_values = model.predict([x_data, t_data])
    sum = 0.0

    for i in range(len(predicted_values)):
        
        cond = np.array(predicted_values[i])[:, 0]
        zero_values = np.zeros((cond.shape[0], ))

        sum += mean_squared_error(zero_values, cond)
    
    return sum

def get_evaluation_value(model, train_data, val_data):
    
    train_loss = calculate_loss(model, train_data[0], train_data[1])
    val_loss = calculate_loss(model, val_data[0], val_data[1])

    return max(train_loss, val_loss)

def generate_train_val_data(t0):

    num_of_samples = 100

    x_data, t_data = np.meshgrid(
        np.linspace(0, 1, num_of_samples),
        np.linspace(t0, 0.5, num_of_samples)
    )
    x_data, t_data = np.array(x_data).reshape(-1, 1), np.array(t_data).reshape(-1, 1)

    x_data, t_data = shuffle(x_data, t_data, random_state = 20)

    return train_test_split(x_data, t_data, test_size = 0.3, shuffle = False)


def stefan_PINN(optimizer, activation, nNeurons, output_activation, nEpoch = 200, earlyStopping = None, alpha = 1.0, t0 = 0.1, TOLX = 0.004, TOLT = 0.002):
    
    learning_rate = 0.002
    batch_size = 512
    
    # Inital
    s0 = alpha * t0

    # Variable definition
    x = sn.Variable('x')
    t = sn.Variable('t')
    u = sn.Field("u")
    s = sn.Field("s")
    
    u = sn.Functional(u, [x, t], nNeurons, activation, output_activation)
    s = sn.Functional(s, [t], nNeurons, activation, output_activation)

    # Diff. equation, heat
    L1 =  diff(u, t) - alpha * diff(u, x, order = 2)

    # Stefan condition
    C1 = (1 / alpha * diff(s, t) + diff(u, x)) * (1 + sign(x - (s - TOLX))) * (1 - sign(x - s))

    # Initial s for t=t0
    C2 = (1 - sign(t - (t0 + TOLT))) * (s - s0)

    # Boundary condition "u" when x=0
    C3 = (1 - sign(x - (0 + TOLX))) * (u - exp(alpha * t))

    # The temperature at the boundary between the phases is 1
    C4 = (1 - sign(x - (s + TOLX))) * (1 + sign(x - s)) * (u - 1)

    x_train, x_val, t_train, t_val = generate_train_val_data(t0)

    """ -------------------------------- MODEL DEFINITION -------------------------------- """
    callbacks = [] 
    if earlyStopping != None:
        callbacks = [earlyStopping]
    
    m = sn.SciModel([x, t], [L1,C1,C2,C3,C4], 'mse', optimizer)
    history = m.train([x_train, t_train], 5 * ['zero'], learning_rate = learning_rate, batch_size = batch_size, epochs = nEpoch, callbacks = [callbacks])
    # m.save_weights('pinn_model_current.hdf5')

    return get_evaluation_value(m, [x_train, t_train], [x_val, t_val])
    




# EPOCH = 3000
EPOCH = 30
ALPHA = 1.0
T0 = 0.1
TOLX = 0.004
TOLT = 0.002

def train_with_hyperopt(params):
    """
    An example train method that calls into MLlib.
    This method is passed to hyperopt.fmin().

    :param params: hyperparameters as a dict. Its structure is consistent with how search space is defined. See below.
    :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
    """
    
    optimizer = params['optimizer']
    activation = params['activation']
    
    # https://github.com/hyperopt/hyperopt/issues/622#issuecomment-698377547 
    # layers_1_unit_1 = hp.uniform('layers_1_unit_1', 16, 32)
    # layers_2_unit_1 = hp.uniform('layers_2_unit_1', 16, 32)
    # layers_2_unit_2 = hp.uniform('layers_2_unit_2', 16, 32)
    # hp.choice('layers', [
    #     [layers_1_unit_1],  # option 1
    #     [layers_2_unit_1, layers_2_unit_2]  # option 2
    # ])
    layers = params['layers']
    layers = [ int(x) for x in layers]
    outputActivation = params['outputActivation']
    # print(layers)
    
    loss = stefan_PINN(optimizer, activation, layers, outputActivation, 
                       nEpoch = EPOCH, 
                       earlyStopping = None, 
                       alpha = ALPHA, 
                       t0 = T0, 
                       TOLX = TOLX, 
                       TOLT = TOLT)
    
    return {'loss': loss, 'status': STATUS_OK}
            
MAX_LAYERS = 10
MIN_UNITS = 16
MAX_UNITS = 32

# Next, define a search space for Hyperopt.
search_space = {
  'optimizer': hp.choice('optimizer', ["Adam", "RMSProp", "Adagrad", "Nadam"]),
  'activation' : hp.choice('activation',  ["tanh", "sigmoid", "selu", "softmax", "relu", "elu"]),
  'outputActivation': hp.choice('outputActivation',  ["linear", "relu"]),
}
layers_choice = []
for layer_count in range(1, MAX_LAYERS + 1):
    options = []
    for i in range(1, layer_count + 1):
        units =  hp.quniform(f'layers_{layer_count}_unit_{i}', MIN_UNITS, MAX_UNITS, 0.9)
        options.append(units)
    layers_choice.append(options)
search_space['layers'] =  hp.choice('layers', layers_choice)





# Select a search algorithm for Hyperopt to use.
algo=tpe.suggest  # Tree of Parzen Estimators, a Bayesian method
best_params = fmin(
    fn=train_with_hyperopt,
    space=search_space,
    algo=algo,
    max_evals=4
)


results = space_eval(search_space, best_params)
results['layers'] = [ int(x) for x in results['layers']]
print(results)

