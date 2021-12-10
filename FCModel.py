#!/usr/bin/python3

import numpy as np
from numpy import genfromtxt
import math

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import wandb

from dataFC import FCDataset
from torch.utils.data import DataLoader

import os

# Logging
use_wandb = True

# Data
num_input_feature = 3
num_tracker = 5
num_output = num_tracker * num_input_feature
sequence_length = 10

# Cuda 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training
num_epochs = 1500
batch_size = 1000
learning_rate_start = 1e-3
learning_rate_end = 1e-4
betas = [0.9, 0.999]

train_data = FCDataset('./data/TrainingData.csv',seq_len=sequence_length, n_input_feat=num_input_feature*num_tracker, n_output=num_output)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=False)

train_data_not_mixed = FCDataset('./data/TrainingDataOCSVM.csv',seq_len=sequence_length, n_input_feat=num_input_feature*num_tracker, n_output=num_output)
train_not_mixed_loader = DataLoader(train_data_not_mixed, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=False)

validation_data = FCDataset('./data/ValidationData.csv',seq_len=sequence_length, n_input_feat=num_input_feature*num_tracker, n_output=num_output)
validationloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=False)

test_data = FCDataset('./data/TestingData.csv',seq_len=sequence_length, n_input_feat=num_input_feature*num_tracker, n_output=num_output)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=False)

test_operating_data = FCDataset('./data/TestingOperatingData.csv',seq_len=sequence_length, n_input_feat=num_input_feature*num_tracker, n_output=num_output)
testoperatingloader = DataLoader(test_operating_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=False)


class TrackerFCNet(nn.Module):
    def __init__(self, device):
        super(TrackerFCNet, self).__init__()
        
        self._device = device
        self.num_epochs = num_epochs
        self.cur_epoch = 0

        hidden_neurons = 100

        layers_backward = []
        layers_backward.append(nn.Linear(sequence_length*num_input_feature*num_tracker, hidden_neurons))
        # layers_backward.append(nn.BatchNorm1d(hidden_neurons))
        layers_backward.append(nn.ReLU())
        layers_backward.append(nn.Linear(hidden_neurons, hidden_neurons))
        # layers_backward.append(nn.BatchNorm1d(hidden_neurons))
        layers_backward.append(nn.ReLU())
        layers_backward.append(nn.Linear(hidden_neurons, num_output))
        
        self.tracker_prediction_network = nn.Sequential(*layers_backward)

        self._optim = optim.Adam(
            self.parameters(),
            lr=learning_rate_start,
            betas=betas
        )
        if use_wandb is True:
            os.environ['WANDB_API_KEY'] = "fead28f24e69ced10a436c3ea4cb26c074ae0dad"
            wandb.init(project="Tracker Residual", tensorboard=False)

    def forward(self, condition, state, input):
        tracker_prediction = self.tracker_prediction_network(torch.cat([condition, state], dim=1))
        return tracker_prediction

    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()

    def fit(self, trainloader, validationloader, print_every=1):
        """
        Train the neural network
        """

        for epoch in range(self.cur_epoch, self.cur_epoch + self.num_epochs):
            print("--------------------------------------------------------")
            print("Training Epoch ", epoch)
            self.cur_epoch += 1

            for param_group in self._optim.param_groups:
                self.learning_rate = learning_rate_start * math.exp(math.log(learning_rate_end/ learning_rate_start) * (epoch / num_epochs))
                param_group['lr'] = self.learning_rate

            train_losses = []
            
            for conditions, states, inputs in trainloader:
                self.train()
                self._optim.zero_grad()

                conditions = conditions.to(self._device)
                states = states.to(self._device)
                inputs = inputs.to(self._device)

                tracker_predictions = self.forward(conditions, states, inputs)
            
                train_loss = nn.L1Loss(reduction='sum')(tracker_predictions, inputs) / inputs.shape[0]

                train_loss.backward()

                self._optim.step()

                train_losses.append(self._to_numpy(train_loss))

            # self.scheduler.step()

            print('Training Loss: ', np.mean(train_losses))

            validation_losses = []
            self.eval()
            for conditions, states, inputs in validationloader:
                conditions = conditions.to(self._device)
                states = states.to(self._device)
                inputs = inputs.to(self._device)

                tracker_predictions = self.forward(conditions, states, inputs)
            
                validation_loss = nn.L1Loss(reduction='sum')(tracker_predictions, inputs) / inputs.shape[0]

                validation_losses.append(self._to_numpy(validation_loss))
            print("Validation Loss: ", np.mean(validation_losses))

            if use_wandb is True:
                wandb_dict = dict()
                wandb_dict['Training Loss'] = np.mean(train_losses)
                wandb_dict['Validation Loss'] =  np.mean(validation_losses)
                wandb_dict['Learning Rate'] =  self.learning_rate
                wandb.log(wandb_dict)

    def save_checkpoint(self):
        """Save model paramers under config['model_path']"""
        model_path = './model/pytorch_model.pt'

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self._optim.state_dict()
        }
        torch.save(checkpoint, model_path)
        if use_wandb is True:
            wandb.save(model_path)

    def restore_model(self, model_path):
        """
        Retore the model parameters
        """
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self._optim.load_state_dict(checkpoint['optimizer_state_dict'])


TrackerFC = TrackerFCNet(device)
TrackerFC.to(device)
TrackerFC.fit(trainloader=trainloader, validationloader=validationloader)
TrackerFC.eval()

TrackerFC.save_checkpoint()
for name, param in TrackerFC.state_dict().items():
    name= name.replace(".","_")
    file_name = "./result/" + name + ".txt"
    np.savetxt(file_name, param.data.cpu())
    if use_wandb is True:
        wandb.save(file_name)

batch_idx = 0
tracker_real_arr = np.empty((0, num_output), float)
tracker_pred_arr = np.empty((0, num_output), float)

for conditions, states, inputs in testloader:
    conditions = conditions.to(TrackerFC._device)
    states = states.to(TrackerFC._device)
    inputs = inputs.to(TrackerFC._device)

    tracker_predictions = TrackerFC.forward(conditions, states, inputs)

    tracker_real_arr = np.append(tracker_real_arr, inputs.cpu().numpy(), axis=0)
    tracker_pred_arr = np.append(tracker_pred_arr, tracker_predictions.cpu().detach().numpy(), axis=0)

    if batch_idx == 0:
        traced_script_module = torch.jit.trace(TrackerFC.to('cpu'), [conditions.cpu(), states.cpu(), inputs.cpu()])
        traced_script_module.save("./model/traced_model.pt")
        TrackerFC.to(device)

    batch_idx = batch_idx+1

np.savetxt('./result/tracker_real.csv',tracker_real_arr)
np.savetxt('./result/tracker_prediction.csv',tracker_pred_arr)


tracker_real_operating_arr = np.empty((0, num_output), float)
tracker_pred_operating_arr = np.empty((0, num_output), float)

for conditions, states, inputs in testoperatingloader:
    conditions = conditions.to(TrackerFC._device)
    states = states.to(TrackerFC._device)
    inputs = inputs.to(TrackerFC._device)

    tracker_predictions = TrackerFC.forward(conditions, states, inputs)

    tracker_real_operating_arr = np.append(tracker_real_operating_arr, inputs.cpu().numpy(), axis=0)
    tracker_pred_operating_arr = np.append(tracker_pred_operating_arr, tracker_predictions.cpu().detach().numpy(), axis=0)

np.savetxt('./result/tracker_operating_real.csv',tracker_real_operating_arr)
np.savetxt('./result/tracker_operating_prediction.csv',tracker_pred_operating_arr)


# Make training data for OCSVM
residual_arr = np.empty((0, num_output), float)

for conditions, states, inputs in train_not_mixed_loader:
    conditions = conditions.to(TrackerFC._device)
    states = states.to(TrackerFC._device)
    inputs = inputs.to(TrackerFC._device)

    backward_dyn_predictions = TrackerFC.forward(conditions, states, inputs)

    residual_arr = np.append(residual_arr, inputs.cpu().numpy() - backward_dyn_predictions.cpu().detach().numpy(), axis=0)

np.savetxt('./OneClassSVM/data/ResidualData.csv',residual_arr)