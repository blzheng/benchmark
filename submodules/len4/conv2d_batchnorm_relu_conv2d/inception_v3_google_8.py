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
        self.conv2d16 = Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d17 = Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x65):
        x66=self.conv2d16(x65)
        x67=self.batchnorm2d16(x66)
        x68=torch.nn.functional.relu(x67,inplace=True)
        x69=self.conv2d17(x68)
        return x69

m = M().eval()
x65 = torch.randn(torch.Size([1, 64, 25, 25]))
start = time.time()
output = m(x65)
end = time.time()
print(end-start)
