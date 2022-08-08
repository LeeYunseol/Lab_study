import torch 
import torch.nn as nn
from statsmodels.tsa.seasonal import STL
import pandas as pd
import numpy as np
from torch.autograd import Variable
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self,  input_size, hidden_size, out_dim, num_layers, seq_length, device):
        super(RNN, self).__init__()
        self.device = device
        self.input_size = input_size # The number of expected features in the input x
        self.hidden_size = hidden_size # The number of features in the hidden state h
        self.out_dim = out_dim # output size
        self.num_layers = num_layers # Number of recurrent layers
        self.seq_length = seq_length # window size

        
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
        # RNN returns output and hidden state
        self.fc = nn.Sequential(nn.Linear(hidden_size * seq_length, 1), nn.Sigmoid())
        # It it is classification distinguishing negative or positive, the activation function should be softmax

        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(device) # 초기 hidden state 설정하기.
        out, hn = self.rnn(x, h0) # out: RNN의 마지막 레이어로부터 나온 output feature 를 반환한다. hn: hidden state를 반환한다.
        print("Output Shape : ", out.shape)
        print("Hidden Shape : ", hn.shape)
        out = out.reshape(out.shape[0], -1) 
        out = self.fc(out)
        
        return out