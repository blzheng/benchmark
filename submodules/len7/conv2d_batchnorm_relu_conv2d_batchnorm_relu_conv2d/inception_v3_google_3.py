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
        self.conv2d22 = Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d23 = Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d24 = Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x76):
        x86=self.conv2d22(x76)
        x87=self.batchnorm2d22(x86)
        x88=torch.nn.functional.relu(x87,inplace=True)
        x89=self.conv2d23(x88)
        x90=self.batchnorm2d23(x89)
        x91=torch.nn.functional.relu(x90,inplace=True)
        x92=self.conv2d24(x91)
        return x92

m = M().eval()
x76 = torch.randn(torch.Size([1, 288, 25, 25]))
start = time.time()
output = m(x76)
end = time.time()
print(end-start)