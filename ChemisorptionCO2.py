import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pandas as pd
import pdfplumber


pdf_path = r'E:\桌面快捷方式\something\My EndNote Library.Data\PDF\2451202095\jz5b01660_si_001.pdf'
BS = 32
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# 第一步.提取数据
pdf = pdfplumber.open(pdf_path)
table = []
for p in pdf.pages[7:14]:
    ctable = p.extract_table()
    table += ctable
df = pd.DataFrame(table[1:])
df.to_csv('datasets.csv')


# 第二步，读取数据
data = pd.read_csv('datasets.csv')

X = torch.Tensor(X_train).to(device)
Y = torch.Tensor(Y_train_orig).to(device, dtype=torch.int64)
x = torch.Tensor(X_test).to(device)
y = torch.Tensor(Y_test_orig).to(device, dtype=torch.int64)

train_data = Data.TensorDataset(X, Y)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BS, shuffle=True)
test_data = Data.TensorDataset(x, y)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BS, shuffle=True)


# 第三步，构建神经网络
class my_nn(nn.module):
    def __init__(self):
        super(my_nn, self).__init__()
        self.hidden1 = nn.Linear(X_train.shape[0], 25)  # 这里是（特征数，第一层hidden神经元素）
        self.hidden2 = nn.Linear(25, 12)
        self.out = nn.Linear(12, 6)

    def forward(self):
        pass

mynet = my_nn().to(device)


def fit(epoch, model, loss_func, optimizer, train_loader, test_loader):
    pass

