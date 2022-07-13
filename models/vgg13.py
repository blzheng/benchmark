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
        self.relu0 = ReLU(inplace=True)
        self.conv2d1 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = ReLU(inplace=True)
        self.maxpool2d0 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d2 = Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = ReLU(inplace=True)
        self.conv2d3 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = ReLU(inplace=True)
        self.maxpool2d1 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d4 = Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5 = ReLU(inplace=True)
        self.maxpool2d2 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d6 = Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu6 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu7 = ReLU(inplace=True)
        self.maxpool2d3 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d8 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu8 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
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
        x1=self.conv2d0(x0)
        x2=self.relu0(x1)
        x3=self.conv2d1(x2)
        x4=self.relu1(x3)
        x5=self.maxpool2d0(x4)
        x6=self.conv2d2(x5)
        x7=self.relu2(x6)
        x8=self.conv2d3(x7)
        x9=self.relu3(x8)
        x10=self.maxpool2d1(x9)
        x11=self.conv2d4(x10)
        x12=self.relu4(x11)
        x13=self.conv2d5(x12)
        x14=self.relu5(x13)
        x15=self.maxpool2d2(x14)
        x16=self.conv2d6(x15)
        x17=self.relu6(x16)
        x18=self.conv2d7(x17)
        x19=self.relu7(x18)
        x20=self.maxpool2d3(x19)
        x21=self.conv2d8(x20)
        x22=self.relu8(x21)
        x23=self.conv2d9(x22)
        x24=self.relu9(x23)
        x25=self.maxpool2d4(x24)
        x26=self.adaptiveavgpool2d0(x25)
        x27=torch.flatten(x26, 1)
        x28=self.linear0(x27)
        x29=self.relu10(x28)
        x30=self.dropout0(x29)
        x31=self.linear1(x30)
        x32=self.relu11(x31)
        x33=self.dropout1(x32)
        x34=self.linear2(x33)
        return [x34]

m = M().eval()
x = torch.randn(1, 3, 224, 224)
start = time.time()
output = m(x)
end = time.time()
print(end-start)
