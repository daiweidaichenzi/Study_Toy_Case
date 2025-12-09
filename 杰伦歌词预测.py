"""
RNN案例，基于杰伦歌词来训练模型，用给定的起始词，结合长度，来进行ai歌词训练

步骤
    获取数据，进行分词，获取词表
    数据预处理，构建数据集
    构建rnn模型
    模型训练
    模型测试

"""
import torch
import torch.nn as nn
import torch.optim as optim
import jieba
from torch.utils.data import DataLoader,Dataset
import time
def build_vocab():
    #定义变量，记录：去重后所有的词，每行文本分词结果
    unique_words,all_words=[],[]
    for line in open('./dataset/jaychou_lyrics.txt','r',encoding='utf-8'):
        words=jieba.lcut(line)
        all_words.append(words)
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    word_count=len(unique_words)
    word_to_index={word:index for index,word in enumerate(unique_words)}
    #歌词文本用词表索引表示
    corpus_idx=[]
    for words in all_words:
        tmp=[]
        for word in words:
            tmp.append(word_to_index[word])
        #在每行词之间，添加空格隔开
        tmp.append(word_to_index[' '])
        corpus_idx.extend(tmp)#用extend，extend把tmp list的每个值都单独添加到corpus_idx中
    #返回唯一词列表，词表，去重后词的数量，歌词文本用词表索引表示
    return unique_words,word_to_index,word_count,corpus_idx

class LyricDataset(Dataset):
        def __init__(self,corpus_idx,num_chars):
            #文档中词的索引
            self.corpus_idx=corpus_idx
            #每个句子中词的个数
            self.num_chars=num_chars
            #文档中词的数量，不去重
            self.word_count=len(corpus_idx)
            #句子数量
            self.number=self.word_count//self.num_chars

            #当使用len（obj），自动调用此方法
        def __len__(self):
            return self.number
        #当使用obj[index]，自动调用该方法
        def __getitem__(self, index):
            #index指的是词的索引，并将其修正索引值到文档的范围中
            #确保索引start在合法范围内，避免越界，start，当前样本的起始索引
            start=min(max(0,index),self.word_count-self.num_chars-1)

            end=start+self.num_chars
            #输入值，从文档中取出从start到end到词索引
            #返回张量形式，可以直接让dataloader加载
            #一般数据转换过程如下
            #数据->张量->dataset->dataloader
            # print()
            return torch.tensor(self.corpus_idx[start:end]),torch.tensor(self.corpus_idx[start+1:end+1])
class LyricModel(nn.Module):
    def __init__(self,word_count):
        super().__init__()
        #语料中词的数量，词向量的维度
        self.emd=nn.Embedding(word_count,128)
        #词向量维度，隐藏层维度，层数
        self.rnn=nn.RNN(128,256,1)
        #输出层(全连接层），特征向量维度（和隐藏维度一直），输出词的数量
        self.output=nn.Linear(256,word_count)#词表中每个词的概率->选概率最大的那个词作为预测结果
    def forward(self,inputs,hidden):
        #初始化词嵌入层

        embd=self.emd(inputs)
        #rnn处理
        output,hidden=self.rnn(embd.transpose(0,1),hidden)
        #全连接,输入内容必须是二维数据
        output=self.output(output.reshape(-1,output.shape[-1]))
        return output,hidden

    def init_hidden(self,bs):#bs:batch_size
        #隐藏层初始化：【网络层数，batch，隐藏层向量维度】
        return torch.zeros(1,bs,256)
def model_train(train_dataset,model):
    unique_words,word_to_index,unique_word_count,corpus_idx=build_vocab()
    criterion= nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    #参一数据集对象，参二批次大小，参三是否打乱数据
    #前期batch_size尽量小一些，后期可以适当调大，因为这样可以让训练次数更多，尽量去选2的倍数
    train_dataloader=DataLoader(train_dataset,batch_size=5,shuffle=True)
    epochs=10
    for epoch in range(epochs):
        start = time.time()
        total_loss=0.0
        batch_num=0
        iter_num=0
        #遍历数据集，会自动调用dataset的getitem方法获取数据,获取到每个样本的数据和标签，根据初始化时的batchsize决定每次拿多少
        for x,y in train_dataloader:
            model.train()
            # 初始化隐藏层,隐藏层
            hidden=model.init_hidden(5)
            print(x.shape)
            output,hidden=model(x,hidden)
            #[批次大小*时间步]
            y=y.transpose(0,1).reshape(-1,)
            # print(y.shape)
            loss=criterion(output,y)#output在这里经过了交叉熵的处理，维度变成了[N,类别数]N是batch size*时间步，这样就可以和y进行对比了，y是每一步的正确类别
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            total_loss+=loss.item()
            iter_num+=1
        # print()
        print(total_loss/iter_num)
        print(f'第{epoch+1}轮训练时间为:{time.time()-start:.2f}秒')
    torch.save(model.state_dict(),'./model/jay_model.pth')

def evaluate(word,sentence_num):
    unique_words,word_to_index,unique_word_count,corpus_idx=build_vocab()
    model=LyricModel(unique_word_count)
    model.load_state_dict(torch.load('./model/jay_model.pth'))
    #获取隐藏层状态
    hidden=model.init_hidden(1)
    #将起始词转换成索引
    word_index=word_to_index[word]
    #定义列表，存放：产生的词的索引
    generate_sentence_index=[word_index]
    for i in range(sentence_num):
        output,hidden=model(torch.tensor([word_index]).unsqueeze(1),hidden)
        #获取输出中，概率最大的词的索引
        word_index=torch.argmax(output,dim=-1).item()
        generate_sentence_index.append(word_index)
    #将索引转换成词
    index_to_word=[unique_words[index]for index in generate_sentence_index]
    for i in generate_sentence_index:
        print(unique_words[i],end='')
    return index_to_word

if __name__ == '__main__':
    #获取数据，进行分词，获取词表
    unique_words,word_to_index,word_count,corpus_idx=build_vocab()
    # print(unique_words)
    # print(word_count)
    # print(corpus_idx)
    # print(word_to_index)
    #构建数据集
    num_chars=32
    dataset=LyricDataset(corpus_idx,num_chars)
    # print(len(dataset))
    # print(dataset[0].shape)
    model=LyricModel(word_count)
    # for name,param in model.named_parameters():
    #     print(name,param.shape)
    # model_train(dataset,model)
    words=evaluate('安静',10)
    # print(words)