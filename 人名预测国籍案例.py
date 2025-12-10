"""
根据人名预测国籍的简单案例
RNN结构
"""
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import string
import pandas as pd
import torch.nn.functional as F
#进度条
from tqdm import tqdm

# 1.todo:获取常用的字符数量：也就是常用的one-hot编码的去重后的词汇的总理n
#string.ascii_letters包含所有的英文字母（大写和小写）
all_letters=string.ascii_letters+"' ,.;"
n_letters=len(all_letters)
batch_size=8
def read_data():
    dataset=pd.read_csv('./dataset/name_classfication.txt',sep='\t',header=None)
    # dataset.info()
    country=dataset.iloc[:,1]
    country_list=pd.unique(country)
    x=dataset.iloc[:,0]
    # my_list_x,my_list_y=[],[]
    # with open('./dataset/name_classfication.txt','r',encoding='utf-8') as f:
    #     for line in f.readlines():
    #         if(len(line)<=5):
    #             continue
    #         x,y=line.strip().split('\t')
    #         my_list_x.append(x)
    #         my_list_y.append(y)
    # print(len(my_list_x))
    return x,country,country_list

#todo：构建dataset类
class NameDataset(Dataset):
    def __init__(self,my_list_x,my_list_y,country_list):
        super().__init__()
        #获取x
        self.my_list_x=my_list_x
        #获取y
        self.my_list_y=my_list_y
        #获取样本数量
        self.sample_len=len(my_list_x)
        self.country_list=country_list
    def __len__(self):
        return self.sample_len
    def __getitem__(self, index):
        index=min(self.sample_len-1,max(0,index))
        x=self.my_list_x[index]
        x=name_padding(x)
        y=self.my_list_y[index]
        #将人名变成one-hot编码形式
        #初始化全0的张量
        tensor_x=name2onehotvec(x)
        tensor_y=torch.tensor(list(self.country_list).index(y),dtype=torch.long)
        return tensor_x,tensor_y
def name2onehotvec(x):
        tensor_x=torch.zeros(len(x),n_letters)
        #遍历人名的每一个字母，进行onehot编码的赋值
        for i,letter in enumerate(x):
            #str.find(letter)返回letter在all_letters中的索引位置
            #list.index(letter)也是返回letter在all_letters中的索引位置
            tensor_x[i][all_letters.find(letter)]=1
        return tensor_x
def name_padding(x):
    if (len(x) < 10):
        x = x + ' ' * (10 - len(x))
    elif (len(x) > 10):
        x = x[:10]
    return x
#todo：实例化dataloader对象
def getDataLoader():
    x,y,country_list=read_data()
    name_dataset=NameDataset(x,y,country_list)
    #会对数据进行增维
    name_dataloader=DataLoader(name_dataset,batch_size=batch_size,shuffle=True)
    return name_dataloader
def eval_model(model,model_name,name_str,country_list):
    name_vector=name2onehotvec(name_padding(name_str))
    name_vector=name_vector.unsqueeze(0)
    k=3
    if(model_name=='LSTM'):
        model.load_state_dict(torch.load('./model/epoch:10 name_:LSTM_model.pth'))
        h0,c0=model.init_hidden(1)
        output,hn=model(name_vector, h0, c0)
        topv,topi=torch.topk(output,dim=-1,k=k)
        for i in range(k):
            print(f'姓名:{name_str},预测国家:{country_list[topi[0][i]]},概率为:{topv[0][i]:.4f}')
        # print(country_list[torch.argmax(model(name_vector,h0,c0)[0],dim=-1)])

    elif(model_name=='GRU'):
        model.load_state_dict(torch.load('./model/epoch:10 name_:GRU_model.pth'))
        h0=model.init_hidden(1)
        output,hn=model(name_vector, h0)

        topv, topi = torch.topk(output, dim=-1, k=k)
        for i in range(k):
            print(f'姓名:{name_str},预测国家:{country_list[topi[0][i]]},概率为:{topv[0][i]:.4f}')
        # print(country_list[torch.argmax(model(name_vector,h0)[0],dim=-1)])

    elif(model_name=='RNN'):
        model.load_state_dict(torch.load('./model/epoch:10 name_:RNN_model.pth'))
        h0=model.init_hidden(1)
        output, hn = model(name_vector, h0)

        topv, topi = torch.topk(output, dim=-1, k=k)
        for i in range(k):
            print(f'姓名:{name_str},预测国家:{country_list[topi[0][i]]},概率为:{topv[0][i]:.4f}')
        # print(country_list[torch.argmax(model(name_vector,h0)[0],dim=-1)])

#todo:构建rnn层
class name_RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super().__init__()
        #input_size:输入的词嵌入维度
        self.input_size=input_size
        #hidden_size:代表RNN模型的隐藏层维度
        self.hidden_size=hidden_size
        #output_size:输出的类别数量
        self.output_size=output_size
        self.num_layers=num_layers
        #构建rnn层,默认情况下batch_first=False
        self.rnn=nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
        self.out=nn.Linear(hidden_size,output_size)
        #定义logsoftmax层
        self.softmax=nn.LogSoftmax(dim=-1)

    def forward(self,x,hidden):
        x=x.transpose(0,1)
        rnn_out,hidden=self.rnn(x,hidden)
        #取rnn_out的最后一个时间步的输出结果
        output=self.out(rnn_out[-1])
        output=self.softmax(output)
        return output,hidden
    def init_hidden(self,bs):
        return torch.zeros(self.num_layers,bs,self.hidden_size)

#todo:构建LSTM模型
class name_LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super().__init__()
        # input_size:输入的词嵌入维度
        self.input_size = input_size
        # hidden_size:代表RNN模型的隐藏层维度
        self.hidden_size = hidden_size
        # output_size:输出的类别数量
        self.output_size = output_size
        self.num_layers = num_layers
        # 构建rnn层,默认情况下batch_first=False
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        # 定义logsoftmax层
        self.softmax = nn.LogSoftmax(dim=-1)
    def forward(self,x,hidden,c):
        x=x.transpose(0,1)
        lstm_out,hidden=self.lstm(x,(hidden,c))
        temp=lstm_out[-1]
        output=self.out(temp)
        output=self.softmax(output)
        return output,hidden
    def init_hidden(self, bs):
         h0=torch.zeros(self.num_layers, bs, self.hidden_size)
         c0=torch.zeros(self.num_layers, bs, self.hidden_size)
         return h0,c0

# todo:构建GRU模型
class name_GRU(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super().__init__()
        #input_size:输入的词嵌入维度
        self.input_size=input_size
        #hidden_size:代表RNN模型的隐藏层维度
        self.hidden_size=hidden_size
        #output_size:输出的类别数量
        self.output_size=output_size
        self.num_layers=num_layers
        #构建rnn层,默认情况下batch_first=False
        self.gru=nn.GRU(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
        self.out=nn.Linear(hidden_size,output_size)
        #定义logsoftmax层
        self.softmax=nn.LogSoftmax(dim=-1)

    def forward(self,x,hidden):
        x=x.transpose(0,1)
        rnn_out,hidden=self.gru(x,hidden)
        #取rnn_out的最后一个时间步的输出结果
        output=self.out(rnn_out[-1])
        output=self.softmax(output)
        return output,hidden
    def init_hidden(self,bs):
        return torch.zeros(self.num_layers,bs,self.hidden_size)

#todo:train函数
#模型训练超参数
mylr=1e-3#一般是10的-3到10的-5次方之间
epoch=10
def model_train(model,model_name):
    dataLoader=getDataLoader()
    #定义损失函数
    #CrossEntropyLoss等价于先进行LogSoftmax再进行NLLloss
    criterion=nn.NLLLoss()
    #定义优化器
    optimizer=optim.Adam(model.parameters(),lr=mylr)
    total_loss = 0.0
    total_iter_num = 0  # 已经开始训练的样本总个数
    total_loss_list = []  # 每隔100个样本计算一下平均损失，画图
    start_time = time.time()
    total_acc_list = []  # 每隔100个样本计算一下平均准确率，画图
    total_acc_num = 0
    total_sample_num=0
    for ep in range(epoch):

        #tqdm进度条加在dataloader遍历的时候
        for x,y in tqdm(dataLoader):
            # print(x.shape)
            if(model_name=='LSTM'):
                h0,c0 = model.init_hidden(bs=x.shape[0])
                output, hidden = model(x, h0,c0)
            else:
                h0=model.init_hidden(bs=x.shape[0])
                output,hidden=model(x,h0)
            loss=criterion(output,y)
            optimizer.zero_grad()
            #反向传播：计算梯度，前向传播时会保留一些激活值，反向传播会把这些激活值读入内存中，避免重复计算
            loss.backward()
            optimizer.step()
            total_iter_num+=1
            total_sample_num+=x.shape[0]
            #新版本不用加item了
            total_loss+=loss.item()
            i_predit_tag=(torch.argmax(output,dim=-1)==y).int().tolist()
            total_acc_num+=sum(i_predit_tag)
            #每一百轮计算一次平均损失和平均准确率
            if(total_iter_num%100==0):
                avg_loss=total_loss/total_iter_num
                total_loss_list.append(avg_loss)
                avg_acc=total_acc_num/total_sample_num
                total_acc_list.append(avg_acc)
            #每训练2000次打印一次日志
            if (total_iter_num%2000==0):
                print(f'Epoch:{ep+1},Iter:{total_iter_num},Loss:{total_loss/total_iter_num:.4f},Acc:{total_acc_num/total_iter_num:.4f},Time:{time.time()-start_time:.2f} seconds')
        #每轮保存一个模型
        torch.save(model.state_dict(),f'./model/epoch:{ep+1} name_:{model_name}_model.pth')
    #将训练的结果保存
    dict1={'total_acc_list':total_acc_list,
           'total_loss_list':total_loss_list,
           'all_time':start_time-time.time()}
    with open(f'./:{model_name}_result.json','w')as f:
        f.write(json.dumps(dict1))
    return total_acc_list,total_loss_list,start_time-time.time()


if __name__ == '__main__':
    # read_data()
    x,y,country_list=read_data()
    dataset=NameDataset(x,y,country_list)
    # print(len(dataset))
    input_size=len(all_letters)
    output_size=(len(country_list))
    num_layers=1
    hidden_size=32
    RNN_model=name_RNN(input_size,hidden_size,output_size,num_layers)
    LSTM_model=name_LSTM(input_size,hidden_size,output_size, num_layers)
    GRU_model=name_GRU(input_size,hidden_size,output_size, num_layers)
    # model_train(LSTM_model,'LSTM')
    eval_model(GRU_model,'GRU','Wanghw',country_list)