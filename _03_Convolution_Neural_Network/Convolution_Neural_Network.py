# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader

# 卷积定义
def conv(in_channels, out_channels, kernel_size, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=padding, bias=False)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size = 3, stride = stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels, kernel_size = 3, stride = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
           residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.conv7x7 = conv(3, 64, kernel_size=3, stride=1) # 原resnet里面使用7x7，由于尺寸过小改为3x3
        self.bn = nn.BatchNorm2d(64)
        # self.max_pool = nn.MaxPool2d(3,2,padding=1) #尺寸过小，最大池化几乎没有作用
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride = 2)
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1):
            downsample = nn.Sequential(
                conv(int(out_channels/2), out_channels, kernel_size=1,stride=stride, padding = 0),
                nn.BatchNorm2d(out_channels)) # 使用1x1卷积块进行下采样
        layers = []
        if (out_channels != 64):  # 其他块需要下采样
            layers.append(block(int(out_channels/2), out_channels, stride, downsample))
        else:                     # 第一块没有下采样，也不需要升维
            layers.append(block(out_channels, out_channels, stride, downsample))

        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv7x7(x)
        out = self.bn(out)
        # out = self.max_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return out


def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    transform = torchvision.transforms.Compose([
    torchvision.transforms.Pad(4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(32),
    torchvision.transforms.ToTensor()])
    # 保持本地训练的数据读取和这里一致
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True, transform=transform)
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=transform)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=50, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=50, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val

def main():
    model = ResNet(ResBlock, [3, 4, 6, 3]) # 若有参数则传入参数 ResNet-34
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir + '/pth/model.pth', map_location=torch.device('cpu')))
    return model
    
