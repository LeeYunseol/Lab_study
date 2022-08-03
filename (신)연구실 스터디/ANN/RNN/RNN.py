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
data_df = pd.DataFrame(data_df, columns =[['Open','High','Low','Close','Volume', 'Adj Close']])


# Make a dataset
X = data_df[['Open','High','Low','Close']].values
y = data_df['Adj Close'].values

# Parameter
window_size = 4
predict_size = 1
split = 2500

X_seq, Y_seq = pp.make_sequence_data(X, y, window_size, predict_size)
X_train = X_seq[:split]
y_train = Y_seq[:split]
X_test = X_seq[split:]
y_test = Y_seq[split:]
'''
# Convert to Tensor 
train_data = pp.TensorData(X_train, y_train)
test_data = pp.TensorData(X_test, y_test)

# Batch
train_loader = DataLoader(train_data, batch_size = 32, shuffle=False)
test_loader = DataLoader(test_data, batch_size = 32, shuffle=False)
'''
train = torch.utils.data.TensorDataset(X_train, y_train)
test = torch.utils.data.TensorDataset(X_test, y_test)

batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
#%%
# Train

input_size = 4 # 4 feature
num_layers = 2
hidden_size = 10
epochs = 200
learning_rate = 0.01
loss_list = [] # list for draw a loss graph

model = model.RNN(input_size = input_size,
                   hidden_size = hidden_size,
                   seq_length = window_size,
                   num_layers = num_layers,
                   out_dim = predict_size,
                   device = device).to(device)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, epochs+1):
    loss_sum = 0
    for batch_idx, samples in enumerate(train_loader):
        
        X_train, y_train = samples
        y_pred = model(X_train.to(device))
    
        loss = criterion(y_pred, y_train.to(device)).to(device)
        loss_sum += loss.item()
    
        # 역전파 수행
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch == 1:
            print('Epoch {:4d}/{} Batch {}/{} Cost: {:.5f}'.format(
            epoch, epochs, batch_idx+1, len(train_loader),
            loss.item()
            ))
            
        if epoch % 20 == 0:
            print('Epoch {:4d}/{} Batch {}/{} Cost: {:.5f}'.format(
            epoch, epochs, batch_idx+1, len(train_loader),
            loss.item()
            ))
    loss_sum = loss_sum / len(train_loader)
    loss_list.append(loss_sum)
#%%
plt.figure(figsize=(20,10))
plt.plot(loss_list)
plt.show()
#%%
def plotting(train_loader, test_loader, actual):
  with torch.no_grad():
    train_pred = []
    test_pred = []

    for data in train_loader:
      seq, target = data
      out = model(seq.to(device))
      train_pred += out.cpu().numpy().tolist()

    for data in test_loader:
      seq, target = data
      out = model(seq.to(device))
      test_pred += out.cpu().numpy().tolist()
      
  total = train_pred + test_pred
  plt.figure(figsize=(20,10))
  plt.plot(np.ones(100)*len(train_pred), np.linspace(0,1,100), '--', linewidth=0.6)
  plt.plot(actual, '--')
  plt.plot(total, 'b', linewidth=0.6)

  plt.legend(['train boundary', 'actual', 'prediction'])
  plt.show()

plotting(train_loader, test_loader, data_df['Adj Close'][window_size:])