import torch
#张量的基础操作
#1.创建张量
a = torch.tensor([[1, 2], [3, 4]]) #直接定义
b = torch.tensor([[5, 6], [7, 8]])
#2.基本运算
c = a + b #逐元素加法
d = torch.matmul(a, b) #矩阵乘法
e = a.view(1, 4) #改变形状为1x4

#3.GPU加速
if torch.cuda.is_available():
    a_gpu = a.cuda() #移动到GPU
    b_gpu = b.cuda()
    c_gpu = a_gpu + b_gpu

#4.结果
print("a:", a)
print("矩阵乘法结果:", d)
print("GPU张量:", c_gpu if torch.cuda.is_available() else "未使用GPU")

#定义极简神经网络
import torch.nn as nn
#1.定义一个2层全连接网络
class MiniNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 2) #输入4维，输出2维
        self.fc2 = nn.Linear(2, 1) #输出1维

    def forward(self, x):
        x = torch.relu(self.fc1(x)) #激活函数
        x = self.fc2(x)
        return x
#2.实例化模型
model = MiniNN()
print("模型结构:", model)
#3.虚拟输入测试
input_data = torch.randn(1, 4) #生成一个样本（1x4）
output = model(input_data)
print("模型输出:", output)

#加载数据集
from torchvision import datasets, transforms

#1.数据预处理
transform = transforms.ToTensor()

#2.下载数据集
train_data = datasets.CIFAR10(
    root='data', train=True, download=True, transform=transform
)
#3.查看数据集信息
print("数据集样本数量:", len(train_data))
print("单个样本形状:", train_data[0][0].shape)