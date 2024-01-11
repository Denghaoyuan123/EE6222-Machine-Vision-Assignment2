import numpy as np
import torch 
import random
import torch.nn as nn
import timm

np.random.seed(2022)
random.seed(2022)
torch.manual_seed(2022)

# Model class
class HARModel(nn.Module):
    
    def __init__(self, model_name, dropout, rnn_hidden_size, rnn_num_layers, num_classes, **params):
        
        super().__init__()
        print("Running on CNN-RNN")
        baseModel = timm.create_model(model_name, **params)        
        num_features = baseModel.fc.in_features
        baseModel.fc = nn.Identity()
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dropout)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)
        
    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        ii = 0
        y = self.baseModel((x[:,ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, ts):
            y = self.baseModel((x[:,ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.fc1(out) 
        return out 