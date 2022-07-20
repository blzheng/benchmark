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
        self.conv2d6 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d6 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.maxpool2d2 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d7 = Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d7 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d8 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d9 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.maxpool2d3 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d10 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d10 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d11 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d12 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)
        self.maxpool2d4 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=(7, 7))
        self.linear0 = Linear(in_features=25088, out_features=4096, bias=True)
        self.relu13 = ReLU(inplace=True)
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.linear1 = Linear(in_features=4096, out_features=4096, bias=True)
        self.relu14 = ReLU(inplace=True)
        self.dropout1 = Dropout(p=0.5, inplace=False)
        self.linear2 = Linear(in_features=4096, out_features=1000, bias=True)

    def forward(self, x):
        x0=x
        x1=self.conv2d0(x0)
        x2=self.batchnorm2d0(x1)
        x3=self.relu0(x2)
        x4=self.conv2d1(x3)
        x5=self.batchnorm2d1(x4)
        x6=self.relu1(x5)
        x7=self.maxpool2d0(x6)
        x8=self.conv2d2(x7)
        x9=self.batchnorm2d2(x8)
        x10=self.relu2(x9)
        x11=self.conv2d3(x10)
        x12=self.batchnorm2d3(x11)
        x13=self.relu3(x12)
        x14=self.maxpool2d1(x13)
        x15=self.conv2d4(x14)
        x16=self.batchnorm2d4(x15)
        x17=self.relu4(x16)
        x18=self.conv2d5(x17)
        x19=self.batchnorm2d5(x18)
        x20=self.relu5(x19)
        x21=self.conv2d6(x20)
        x22=self.batchnorm2d6(x21)
        x23=self.relu6(x22)
        x24=self.maxpool2d2(x23)
        x25=self.conv2d7(x24)
        x26=self.batchnorm2d7(x25)
        x27=self.relu7(x26)
        x28=self.conv2d8(x27)
        x29=self.batchnorm2d8(x28)
        x30=self.relu8(x29)
        x31=self.conv2d9(x30)
        x32=self.batchnorm2d9(x31)
        x33=self.relu9(x32)
        x34=self.maxpool2d3(x33)
        x35=self.conv2d10(x34)
        x36=self.batchnorm2d10(x35)
        x37=self.relu10(x36)
        x38=self.conv2d11(x37)
        x39=self.batchnorm2d11(x38)
        x40=self.relu11(x39)
        x41=self.conv2d12(x40)
        x42=self.batchnorm2d12(x41)
        x43=self.relu12(x42)
        x44=self.maxpool2d4(x43)
        x45=self.adaptiveavgpool2d0(x44)
        x46=torch.flatten(x45, 1)
        x47=self.linear0(x46)
        x48=self.relu13(x47)
        x49=self.dropout0(x48)
        x50=self.linear1(x49)
        x51=self.relu14(x50)
        x52=self.dropout1(x51)
        x53=self.linear2(x52)

m = M().eval()
x = torch.randn(1, 3, 224, 224)
start = time.time()
output = m(x)
end = time.time()
print(end-start)