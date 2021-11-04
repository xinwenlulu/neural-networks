import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from torchsummary import summary
import pandas as pd

import part1_nn_lib as lib


seed = 60012
rg = default_rng(seed)
weights = np.array([4, 2.5, 1.5])

n_samples = 1000
x = rg.random((n_samples, 2))*10
x = np.hstack((x, np.ones((n_samples, 1))))
y = np.matmul(x, weights)
noise = rg.standard_normal(y.shape)
y = y + noise


x_train = x[:800, :2]
y_train = y[:800]
x_val = x[800:, :2]
y_val = y[800:]


input_dim = 2
neurons = [1]
activations = ["relu"]
net = lib.MultiLayerNetwork(input_dim, neurons, activations)


prep_input = lib.Preprocessor(x_train)
x_train_pre = prep_input.apply(x_train)
x_val_pre = prep_input.apply(x_val)

trainer = lib.Trainer(
        network=net,
        batch_size=20,
        nb_epoch=1000,#1000
        learning_rate=0.001,
        loss_fun="mse",
        shuffle_flag=True,
)

trainer.train(x_train_pre, y_train)
print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

preds = net(x_val_pre).squeeze()
targets = y_val.squeeze()

