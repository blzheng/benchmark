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
        self.conv2d0 = Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d0 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu0 = ReLU(inplace=True)
        self.conv2d1 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)
        self.maxpool2d0 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d2 = Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d2 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = ReLU(inplace=True)
        self.conv2d3 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d3 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.maxpool2d1 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d4 = Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d4 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d5 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.maxpool2d2 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d6 = Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d6 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d7 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.maxpool2d3 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d8 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d8 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d9 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.maxpool2d4 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=(7, 7))
        self.linear0 = Linear(in_features=25088, out_features=4096, bias=True)
        self.relu10 = ReLU(inplace=True)
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.linear1 = Linear(in_features=4096, out_features=4096, bias=True)
        self.relu11 = ReLU(inplace=True)
        self.dropout1 = Dropout(p=0.5, inplace=False)
        self.linear2 = Linear(in_features=4096, out_features=1000, bias=True)

    def forward(self, x):
        x0=x
        print('x0: {}'.format(x0.shape))
        x1=self.conv2d0(x0)
        print('x1: {}'.format(x1.shape))
        x2=self.batchnorm2d0(x1)
        print('x2: {}'.format(x2.shape))
        x3=self.relu0(x2)
        print('x3: {}'.format(x3.shape))
        x4=self.conv2d1(x3)
        print('x4: {}'.format(x4.shape))
        x5=self.batchnorm2d1(x4)
        print('x5: {}'.format(x5.shape))
        x6=self.relu1(x5)
        print('x6: {}'.format(x6.shape))
        x7=self.maxpool2d0(x6)
        print('x7: {}'.format(x7.shape))
        x8=self.conv2d2(x7)
        print('x8: {}'.format(x8.shape))
        x9=self.batchnorm2d2(x8)
        print('x9: {}'.format(x9.shape))
        x10=self.relu2(x9)
        print('x10: {}'.format(x10.shape))
        x11=self.conv2d3(x10)
        print('x11: {}'.format(x11.shape))
        x12=self.batchnorm2d3(x11)
        print('x12: {}'.format(x12.shape))
        x13=self.relu3(x12)
        print('x13: {}'.format(x13.shape))
        x14=self.maxpool2d1(x13)
        print('x14: {}'.format(x14.shape))
        x15=self.conv2d4(x14)
        print('x15: {}'.format(x15.shape))
        x16=self.batchnorm2d4(x15)
        print('x16: {}'.format(x16.shape))
        x17=self.relu4(x16)
        print('x17: {}'.format(x17.shape))
        x18=self.conv2d5(x17)
        print('x18: {}'.format(x18.shape))
        x19=self.batchnorm2d5(x18)
        print('x19: {}'.format(x19.shape))
        x20=self.relu5(x19)
        print('x20: {}'.format(x20.shape))
        x21=self.maxpool2d2(x20)
        print('x21: {}'.format(x21.shape))
        x22=self.conv2d6(x21)
        print('x22: {}'.format(x22.shape))
        x23=self.batchnorm2d6(x22)
        print('x23: {}'.format(x23.shape))
        x24=self.relu6(x23)
        print('x24: {}'.format(x24.shape))
        x25=self.conv2d7(x24)
        print('x25: {}'.format(x25.shape))
        x26=self.batchnorm2d7(x25)
        print('x26: {}'.format(x26.shape))
        x27=self.relu7(x26)
        print('x27: {}'.format(x27.shape))
        x28=self.maxpool2d3(x27)
        print('x28: {}'.format(x28.shape))
        x29=self.conv2d8(x28)
        print('x29: {}'.format(x29.shape))
        x30=self.batchnorm2d8(x29)
        print('x30: {}'.format(x30.shape))
        x31=self.relu8(x30)
        print('x31: {}'.format(x31.shape))
        x32=self.conv2d9(x31)
        print('x32: {}'.format(x32.shape))
        x33=self.batchnorm2d9(x32)
        print('x33: {}'.format(x33.shape))
        x34=self.relu9(x33)
        print('x34: {}'.format(x34.shape))
        x35=self.maxpool2d4(x34)
        print('x35: {}'.format(x35.shape))
        x36=self.adaptiveavgpool2d0(x35)
        print('x36: {}'.format(x36.shape))
        x37=torch.flatten(x36, 1)
        print('x37: {}'.format(x37.shape))
        x38=self.linear0(x37)
        print('x38: {}'.format(x38.shape))
        x39=self.relu10(x38)
        print('x39: {}'.format(x39.shape))
        x40=self.dropout0(x39)
        print('x40: {}'.format(x40.shape))
        x41=self.linear1(x40)
        print('x41: {}'.format(x41.shape))
        x42=self.relu11(x41)
        print('x42: {}'.format(x42.shape))
        x43=self.dropout1(x42)
        print('x43: {}'.format(x43.shape))
        x44=self.linear2(x43)
        print('x44: {}'.format(x44.shape))

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
