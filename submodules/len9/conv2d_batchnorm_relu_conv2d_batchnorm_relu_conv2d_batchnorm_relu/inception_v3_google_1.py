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
        self.conv2d8 = Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d9 = Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d10 = Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x30):
        x40=self.conv2d8(x30)
        x41=self.batchnorm2d8(x40)
        x42=torch.nn.functional.relu(x41,inplace=True)
        x43=self.conv2d9(x42)
        x44=self.batchnorm2d9(x43)
        x45=torch.nn.functional.relu(x44,inplace=True)
        x46=self.conv2d10(x45)
        x47=self.batchnorm2d10(x46)
        x48=torch.nn.functional.relu(x47,inplace=True)
        return x48

m = M().eval()
x30 = torch.randn(torch.Size([1, 192, 25, 25]))
start = time.time()
output = m(x30)
end = time.time()
print(end-start)
