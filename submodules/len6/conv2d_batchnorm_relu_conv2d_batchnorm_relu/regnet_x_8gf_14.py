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
        self.conv2d46 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d47 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu44 = ReLU(inplace=True)

    def forward(self, x149):
        x150=self.conv2d46(x149)
        x151=self.batchnorm2d46(x150)
        x152=self.relu43(x151)
        x153=self.conv2d47(x152)
        x154=self.batchnorm2d47(x153)
        x155=self.relu44(x154)
        return x155

m = M().eval()
x149 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x149)
end = time.time()
print(end-start)
