import torch
import torchvision.datasets as dset
from torch.utils.data import TensorDataset,DataLoader, Dataset
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)#들어온 데이터를 텐서로 
        self.y_data = torch.FloatTensor(y_data)  #들어온 데이터를 텐서로 
        self.len = self.y_data.shape[0]
        
    def __getitem__ (self, index): #
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
def make_sequence_data(feature, label, window_size, predict_size):
    feature_list = []      # 생성될 feature list
    label_list = []        # 생성될 label list
    for i in range(len(feature)-window_size-predict_size+1):
        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size:i+window_size+predict_size]) 
    return np.array(feature_list), np.array(label_list)