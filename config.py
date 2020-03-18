from torch import nn


activation = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "tanh": nn.Tanh()
}

parameters = {
    "EPOCHS" : 20,
    "LEARNING_RATE": 0.01 
}