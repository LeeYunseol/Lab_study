import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
import src.preprocess as pp
import src.model as model
from scipy import stats

# GPU 설정 및 랜덤 시드 고정

data = pd.read_csv('samsung.csv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device : {}".format(device))
#%%
data.info()
#%%
# Box Plot
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
fig.suptitle('Box Plot', fontsize=16)
sns.boxplot(data['High'], ax=ax[0,0])
sns.boxplot(data['Low'], ax=ax[0,1])
sns.boxplot(data['Open'], ax=ax[0,2])
sns.boxplot(data['Close'], ax=ax[1,0])
sns.boxplot(data['Volume'], ax=ax[1,1])
sns.boxplot(data['Adj Close'], ax=ax[1,2])
#%%
# Correlation
plt.figure(figsize=(28,14))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
sns.heatmap(data.corr(), cmap='Reds', annot=True, annot_kws={'size':12})
plt.title('Correlation Matrix', fontsize=30)
#%%
fig = plt.figure(figsize=(20,15))                 
plt.title('Scatter')
index =1 
columns = list(data.columns)
columns.remove('Date')
for column in columns :
    ax = fig.add_subplot(3, 2, index)
    sns.scatterplot(data[column], y = data['Adj Close'])
    index+=1
plt.show()
#%%
# Before Log Transform
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
fig.suptitle('Before Log Transform', fontsize=16)
sns.distplot(data['High'], ax=ax[0,0], fit = stats.norm)
sns.distplot(data['Low'], ax=ax[0,1], fit = stats.norm)
sns.distplot(data['Open'], ax=ax[0,2], fit = stats.norm)
sns.distplot(data['Close'], ax=ax[1,0], fit = stats.norm)
sns.distplot(data['Volume'], ax=ax[1,1], fit = stats.norm)
sns.distplot(data['Adj Close'], ax=ax[1,2], fit = stats.norm)
#%%
# After Log Transform
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
fig.suptitle('After Log Transform', fontsize=16)
sns.distplot(np.log1p(data['High']), ax=ax[0,0], fit = stats.norm)
sns.distplot(np.log1p(data['Low']), ax=ax[0,1], fit = stats.norm)
sns.distplot(np.log1p(data['Open']), ax=ax[0,2], fit = stats.norm)
sns.distplot(np.log1p(data['Close']), ax=ax[1,0], fit = stats.norm)
sns.distplot(np.log1p(data['Volume']), ax=ax[1,1], fit = stats.norm)
sns.distplot(np.log1p(data['Adj Close']), ax=ax[1,2], fit = stats.norm)

#%%
# Prepeocess
'''
data['High'] = np.log1p(data['High'])
data['Low'] = np.log1p(data['Low'])
data['Open'] = np.log1p(data['Open'])
data['Close'] = np.log1p(data['Close'])
data['Volume'] = np.log1p(data['Volume'])
data['Adj Close'] = np.log1p(data['Adj Close'])
'''
#%%
# MinMaxScaler 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_df = scaler.fit_transform(data[['Open','High','Low','Close','Volume', 'Adj Close']])
data_df = pd.DataFrame(data_df, columns =[['Open','High','Low','Close','Volume', 'Adj Close']] )


# Make a dataset
X = data_df[['Open','High','Low','Close']].values
y = data_df['Adj Close'].values

# Parameter
window_size = 4
predict_size = 1
split = 2500

def seq_data(x, y, sequence_length):
  
  x_seq = []
  y_seq = []
  for i in range(len(x) - sequence_length):
    x_seq.append(x[i: i+sequence_length])
    y_seq.append(y[i+sequence_length])

  return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view([-1, 1]) # float형 tensor로 변형, gpu사용가능하게 .to(device)를 사용.

x_seq, y_seq = seq_data(X, y, window_size)

x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]
x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]
print(x_train_seq.size(), y_train_seq.size())
print(x_test_seq.size(), y_test_seq.size())

train = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)

batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

input_size = 4
num_layers = 2
hidden_size = 10

class VanillaRNN(nn.Module):

  def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
    super(VanillaRNN, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 1), nn.Sigmoid())

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device) # 초기 hidden state 설정하기.
    out, _ = self.rnn(x, h0) # out: RNN의 마지막 레이어로부터 나온 output feature 를 반환한다. hn: hidden state를 반환한다.
    out = out.reshape(out.shape[0], -1) # many to many 전략
    out = self.fc(out)
    return out
model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   sequence_length=window_size,
                   num_layers=num_layers,
                   device=device).to(device)

criterion = nn.MSELoss()

lr = 0.01
num_epochs = 200
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(train_loader)

for epoch in range(num_epochs):
  running_loss = 0.0

  for data in train_loader:

    seq, target = data # 배치 데이터.
    out = model(seq)   # 모델에 넣고,
    loss = criterion(out, target) # output 가지고 loss 구하고,

    optimizer.zero_grad() # 
    loss.backward() # loss가 최소가 되게하는 
    optimizer.step() # 가중치 업데이트 해주고,
    running_loss += loss.item() # 한 배치의 loss 더해주고,

  loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
  if epoch % 100 == 0:
    print('[epoch: %d] loss: %.4f'%(epoch, running_loss/n))
#%%
plt.figure(figsize=(20,10))
plt.plot(loss_graph)
plt.show()
#%%

def plotting(train_loader, test_loader, actual):
    
    with torch.no_grad():
        train_pred = []
        test_pred = []

        for data in train_loader:
            seq, target = data
            out = model(seq)
            train_pred += out.cpu().numpy().tolist()
    
        for data in test_loader:
            seq, target = data
            out = model(seq)
            test_pred += out.cpu().numpy().tolist()
      
    total = train_pred + test_pred
    plt.figure(figsize=(20,10))
    plt.plot(np.ones(100)*len(train_pred), np.linspace(0,1,100), '--', linewidth=0.6)
    plt.plot(actual, '--')
    plt.plot(total, 'b', linewidth=0.6)
      
    plt.legend(['train boundary', 'actual', 'prediction'])
    plt.show()
#%%
plotting(train_loader, test_loader, data_df['Adj Close'][window_size:])