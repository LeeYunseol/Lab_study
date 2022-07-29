import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
# GPU 설정 및 랜덤 시드 고정

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device : {}".format(device))

# ANN 모델 생성
class Model(nn.Module):
    # torch.manual_seed(2022)
    # 총 5개의 feature를 input으로 사용해서 output 하나가 나오는 형태, 은닉층은 총 3개, 각 layer 당 10개
    def __init__(self, in_features=5, h1 = 10, h2 = 10, h3 = 10 , out_features=1) :
        print("모델 생성")
        super(Model, self).__init__()
        # input -> h1(첫 번째 hidden layer)
        self.fc1 = nn.Linear(in_features, h1)
        #print("첫 번째 Hidden Layer Shape : {}".format(self.fc1))
        # h1 -> h2
        #self.fc2 = nn.Linear(h1, h2)
        #print("두 번째 Hidden Layer Shape : {}".format(self.fc2))
        # h2 -> h3
        #self.fc3 = nn.Linear(h2, h3)
        #print("세 번째 Hidden Layer Shape : {}".format(self.fc3))
        # h3 -> output
        self.out = nn.Linear(h3, out_features)
        
        # 입력 -> 은닉층 1 -> 은닉층 2 -> 은닉층 3 ->  출력
    
    #순전파
    def forward(self, x):
        # print("Input Shape : {}".format(x.shape))
        x = F.relu(self.fc1(x))
        # print("Output Shape After First Hidden Layer : {}".format(x.shape))
        #x = F.relu(self.fc2(x))
        # print("Output Shape After Second Hidden Layer : {}".format(x.shape))
        #x = F.relu(self.fc3(x))
        # print("Output Shape After Third Hidden Layer : {}".format(x.shape))
        x = self.out(x)
        # print("Output Shape : {}".format(x.shape))
        return x
#%%
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

# Load Data
data = pd.read_csv('samsung.csv')

X = data.drop(['Date', 'Adj Close'], axis=1) 
y = data['Adj Close']


# 데이터 프레임 -> numpy array 형태로 추출
# 학습을 위해서는 DataFrame -> numpy array -> Tensor로 변환
X=X.values
y=y.values    
    
from sklearn.model_selection import train_test_split

# train/test 비율 8 : 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022)

# 위에서 설명한 데이터 텐서화
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# Train dataset 만들기
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last= False) # 미니 배치 크기와 데이터 크기가 다를 경우 마지막 단위를 어떻게 처리할지는 drop_last 인수로 설정

# 모델 선언
model = Model()

# 손실함수를 MSE로 정의
criterion = torch.nn.MSELoss().to(device)

# 최적화 함수 정의
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Learning rate 차이 보여주기!!!!
#%%
# 학습

epochs = 100 # 훈련 횟수 100번
loss_list = [] # loss를 담을 리스트

for epoch in range(1, epochs+1):
    loss_sum = 0
    for batch_idx, samples in enumerate(loader):
        
        X_train, y_train = samples
        
        model.train()
        y_pred = model(X_train)
        y_pred = y_pred.squeeze()
    
        loss = criterion(y_pred, y_train)
        loss_sum += loss.item()
    
        # 역전파 수행
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch == 1:
            print('Epoch {:4d}/{} Batch {}/{} Cost: {:.0f}'.format(
            epoch, epochs, batch_idx+1, len(loader),
            loss.item()
            ))
            
        if epoch % 20 == 0:
            print('Epoch {:4d}/{} Batch {}/{} Cost: {:.0f}'.format(
            epoch, epochs, batch_idx+1, len(loader),
            loss.item()
            ))
    loss_list.append(loss_sum)
#%%
# 시각화
plt.plot(range(epochs), loss_list)
plt.ylabel('loss')
plt.xlabel('Epoch')
#%%
# Result Visualization
plt.figure(figsize=(10,10))
plt.plot(y_test.detach().cpu().numpy(),  label="Original Data")
plt.plot(model(X_test).detach().cpu().numpy(), label="Model Output")
plt.legend()
plt.show()

#%%
# RMSE 출력
from sklearn.metrics import mean_squared_error
from math import sqrt

RMSE = sqrt(mean_squared_error(y_test.detach().cpu().numpy(), model(X_test).detach().cpu().numpy()))

print("RMSE : {}".format(RMSE))