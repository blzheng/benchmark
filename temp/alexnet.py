import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator
import sys
import os

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d0 = Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.relu0 = ReLU(inplace=True)
        self.maxpool2d0 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d1 = Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.relu1 = ReLU(inplace=True)
        self.maxpool2d1 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d2 = Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = ReLU(inplace=True)
        self.conv2d3 = Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = ReLU(inplace=True)
        self.conv2d4 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = ReLU(inplace=True)
        self.maxpool2d2 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=(6, 6))
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.linear0 = Linear(in_features=9216, out_features=4096, bias=True)
        self.relu5 = ReLU(inplace=True)
        self.dropout1 = Dropout(p=0.5, inplace=False)
        self.linear1 = Linear(in_features=4096, out_features=4096, bias=True)
        self.relu6 = ReLU(inplace=True)
        self.linear2 = Linear(in_features=4096, out_features=1000, bias=True)

    def forward(self, x):
        x0=x
        print('x0: {}'.format(x0.shape))
        x1=self.conv2d0(x0)
        print('x1: {}'.format(x1.shape))
        x2=self.relu0(x1)
        print('x2: {}'.format(x2.shape))
        x3=self.maxpool2d0(x2)
        print('x3: {}'.format(x3.shape))
        x4=self.conv2d1(x3)
        print('x4: {}'.format(x4.shape))
        x5=self.relu1(x4)
        print('x5: {}'.format(x5.shape))
        x6=self.maxpool2d1(x5)
        print('x6: {}'.format(x6.shape))
        x7=self.conv2d2(x6)
        print('x7: {}'.format(x7.shape))
        x8=self.relu2(x7)
        print('x8: {}'.format(x8.shape))
        x9=self.conv2d3(x8)
        print('x9: {}'.format(x9.shape))
        x10=self.relu3(x9)
        print('x10: {}'.format(x10.shape))
        x11=self.conv2d4(x10)
        print('x11: {}'.format(x11.shape))
        x12=self.relu4(x11)
        print('x12: {}'.format(x12.shape))
        x13=self.maxpool2d2(x12)
        print('x13: {}'.format(x13.shape))
        x14=self.adaptiveavgpool2d0(x13)
        print('x14: {}'.format(x14.shape))
        x15=torch.flatten(x14, 1)
        print('x15: {}'.format(x15.shape))
        x16=self.dropout0(x15)
        print('x16: {}'.format(x16.shape))
        x17=self.linear0(x16)
        print('x17: {}'.format(x17.shape))
        x18=self.relu5(x17)
        print('x18: {}'.format(x18.shape))
        x19=self.dropout1(x18)
        print('x19: {}'.format(x19.shape))
        x20=self.linear1(x19)
        print('x20: {}'.format(x20.shape))
        x21=self.relu6(x20)
        print('x21: {}'.format(x21.shape))
        x22=self.linear2(x21)
        print('x22: {}'.format(x22.shape))

m = M().eval()
CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x = torch.randn(batch_size, 3, 224, 224)
start_time=time.time()
for i in range(10):
    output = m(x)
total_iter_time = time.time() - start_time
Throughput = batch_size * 10 / total_iter_time
file_current = os.path.basename(__file__)
print(file_current,',',BS,',',Throughput) 
