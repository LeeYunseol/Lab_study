# 이해가 되지 않아.. 아직 미완성입니다..

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader, Dataset
import random
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device : ", device)

data = pd.read_csv('samsung.csv')
plt.figure(figsize=(20,5))
plt.plot(range(len(data)), data["Adj Close"])

#%%
# 함수

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
    feature_list = []      
    label_list = []       
    for i in range(len(feature) - window_size - predict_size + 1):
        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i + window_size : i + window_size + predict_size]) 
        
    return np.array(feature_list), np.array(label_list)
#%%

# 일단은 시각화를 위해 전체 데이터셋에 대해서 MinMaxScaler 적용
data = data.drop(['Date'], axis = 1)
scaler = MinMaxScaler()
data_df = scaler.fit_transform(data)
data_df = pd.DataFrame(data = data_df, columns= data.columns)

# Train, Test data 확인
X = data_df[['High', 'Low', 'Open', 'Close']]
y = data_df[['Adj Close']]



# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022, shuffle=False)
# Make a dataset
#X_train = X_train[['Open','High','Low','Close']].values
y_train = y_train['Adj Close'].values
#X_test = X_test[['Open','High','Low','Close']].values
y_test = y_test['Adj Close'].values

# Parameter
window_size = 4
predict_size = 4


X_train, y_train = make_sequence_data(X_train, y_train, window_size, predict_size)
X_test, y_test = make_sequence_data(X_test, y_test, window_size, predict_size)

plt.figure(figsize=(20,30))
fig, axs =plt.subplots(3,figsize=(12,15))
axs[0].plot(y.values)
axs[0].title.set_text('Original Time Series')
axs[0].set_xlim(0,len(y))
axs[0].set_ylim(0,1)
axs[1].plot(y_train,color="red")
axs[1].title.set_text('Train Data')
axs[1].set_xlim(0,len(y))
temp = np.full(len(y_train)*4, 0.328).reshape(-1,4)
axs[1].set_ylim(0,1)
axs[2].plot(np.concatenate([temp, y_test], 0),color='black')
axs[2].title.set_text('Test Data')
axs[2].set_xlim(0,len(y))
axs[2].set_ylim(0,1)
#%%
# LSTM Encoder
# Encoder : input을 통해 decoder에 전달할 hidden state 생성
class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)

    def forward(self, x_input):
        #x: (batch_size, seq, feature_num)
        
        # x_input shape : 32 4 5
        #hidden / cell state 초기화 
        # h_0: (numlayers * numdirections, batch, hiddensize)
        h_0 = Variable(torch.zeros(self.num_layers, x_input.size(0), self.hidden_size)).to(device) #은닉상태 0 초기화
        c_0 = Variable(torch.zeros(self.num_layers, x_input.size(0), self.hidden_size)).to(device) #셀 상태 0 초기화 
        
        outputs, (hidden, cell) = self.lstm(x_input, (h_0, c_0))
        
        #Encoder의 마지막 hidden state만 디코더로 전달 => Context vector [num_layers, batch_size, enc_hid_dim]
        return hidden, cell

#%%
# LSTM Decoder
# Decoder : input의 마지막 값과 encoder에서 받은 hidden state를 이용하여 한 개의 값을 예측
class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, out_dim,  num_layers = 1): # 5 16
        super(lstm_decoder, self).__init__()
        self.input_size = 1 # 5
        self.hidden_size = hidden_size # 16
        self.num_layers = num_layers
        self.out_dim = out_dim
        
        self.lstm = nn.LSTM(input_size = 1, hidden_size = hidden_size,num_layers = num_layers, batch_first=True)
        # Decoder에서 새로 추가된 부분
        # 설명의 다음 단어를 예측한다면 활성화 함수로 softmax 함수가 타당하지만 시계열 예측이기 때문에 여기에서는 linear 사용
        self.linear = nn.Linear(hidden_size, out_dim)           
    
    def forward(self, input, hidden, cell):
        ##input: Decoder의 시작 신호로는 Encoder에 주입했던 마지막 시점의 예측 변수가 들어가야 함 [1, size, 예측 변수 개수] 
        ##hidden, cell: Encoder에서 나온 마지막 시점의 hidden / cell state가 Decoder에 주입됨
        # print("input shape : ", input.shape)
        # input shape : torch.Size([32, 4, 1])
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        #print("output of decoder : ", output.shape)
        # output of decoder :  torch.Size([32, 4, 16])
        # 0 차원의 1을 삭제 
        prediction = self.linear(output)
        print(prediction.shape)
        # prediction = prediction.unsqueeze(0)
        print(prediction.shape)        
        
        return prediction, hidden, cell

#%%
# Encoder & Decoder

class lstm_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, out_dim):
        super(lstm_encoder_decoder, self).__init__()

        self.input_size = input_size # 5 -> 5 features
        self.hidden_size = hidden_size # 16
        self.out_dim = out_dim
        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size, out_dim = out_dim)

    def forward(self, inputs, targets, target_len, teacher_forcing_ratio):
        # print("input of decoder shape : ", inputs.shape)
        # input of decoder shape : torch.Size([32, 4, 5]) => 이거는 decoder에 들어가는 input
        batch_size = inputs.shape[1] # 32
        input_size = inputs.shape[2] # 5

        outputs = torch.zeros(batch_size, self.out_dim, target_len) # 32 4 5

        # Encoder에서 inputs을 넣고 hidden을 뽑아내고
        hidden, cell = self.encoder(inputs)
        # print("hidden state after encoder : ", hidden.shape)
        # print("cell state after encoder : ", cell.shape)
        # hidden state after encoder : torch.Size([1, 32, 16])
        # cell state after encoder :  torch.Size([1, 32, 16])
        decoder_input = inputs[:, : , -1:] # decoder_input : 32 4 1
        
        #원하는 길이가 될 때까지 decoder를 실행한다.
        for t in range(target_len): 
            out, hidden, cell = self.decoder(decoder_input, hidden, cell) 
            # print("output of decoder shape :", out.shape)
            # output of decoder shape : torch.Size([32, 4, 1])
            print("섹스", out.shape)
            # 32, 4
            # teacher forcing을 구현한다.
            # teacher forcing에 해당하면 다음 인풋값으로는 예측한 값이 아니라 실제 값을 사용한다.
            if random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :]
                
            else:
                decoder_input = out
            outputs[:,t,:] = out
            
        return outputs
	
    # 편의성을 위해 예측해주는 함수도 생성한다.
    def predict(self, inputs, target_len):
        self.eval()
        inputs = inputs.unsqueeze(0)
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]
        outputs = torch.zeros(batch_size, target_len, input_size)
        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:,-1, :]
        out, hidden = self.decoder(decoder_input, hidden)
        out =  out.squeeze(1)

        return out
#%%

X = data[['High', 'Low', 'Open', 'Close', 'Adj Close']]
y = data[['Adj Close']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022, shuffle=False)

scaler_for_x = MinMaxScaler()
scaler_for_x.fit(X_train)
X_train_df = scaler_for_x.transform(X_train)
X_train = pd.DataFrame(data=X_train_df, columns = X_train.columns)

X_test_df = scaler_for_x.transform(X_test)
X_test = pd.DataFrame(data = X_test_df, columns = X_train.columns)

scaler_for_y = MinMaxScaler()
scaler_for_y.fit(y_train)
y_train_df = scaler_for_y.transform(y_train)
y_train = pd.DataFrame(data=y_train_df, columns = y_train.columns)

y_test_df = scaler_for_y.transform(y_test)
y_test = pd.DataFrame(data = y_test_df, columns = y_train.columns)


# Make a dataset
X_train = X_train[['Open','High','Low','Close', 'Adj Close']].values
y_train = y_train['Adj Close'].values

X_test = X_test[['Open','High','Low','Close', 'Adj Close']].values
y_test = y_test['Adj Close'].values

# Parameter
window_size = 4
predict_size = 4


X_train, y_train = make_sequence_data(X_train, y_train, window_size, predict_size)
X_test, y_test = make_sequence_data(X_test, y_test, window_size, predict_size)
# 후에 시각화를 위해 저장
actual = y_train.tolist() + y_test.tolist()

# Convert to Tensor 
train_data = TensorData(X_train, y_train)
test_data = TensorData(X_test, y_test)

# Batch
train_loader = DataLoader(train_data, batch_size = 32, shuffle=False)
test_loader = DataLoader(test_data, batch_size = 32, shuffle=False)


model = lstm_encoder_decoder(input_size=5, hidden_size=16, out_dim = 4).to(device)
learning_rate=0.01
epoch = 200
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()
from tqdm import tqdm

model.train()
with tqdm(range(epoch)) as tr:
    for i in tr:
        total_loss = 0.0
        for x,y in train_loader:
            optimizer.zero_grad()
            y= y.unsqueeze(dim = 2)
            print("x shape : ", x.shape)
            print("y. shape : ", y.shape)
            # x : [batch size, seq_length, input_size]
            # y : [batch size, seq_length]
            # x : torch.Size([32, 4, 5])
            # y : torch.Size([32, 4])
            
            x = x.to(device).float()
            y = y.to(device).float()
            
            output = model(x, y, 1, 0.6).to(device) # 세 번째 parameter는 target len

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()
        tr.set_postfix(loss="{0:.5f}".format(total_loss/len(train_loader)))
#%%
with torch.no_grad():
    test_pred = []

for data in test_loader :
    x, y = data
    print(x.shape)
    prediction = model.predict(x.to(device).float(), 1)
    prediction = prediction.cpu().detach().numpy()

