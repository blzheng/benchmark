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
        self.conv2d18 = Conv2d(448, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)

    def forward(self, x55):
        x56=self.conv2d18(x55)
        x57=self.batchnorm2d12(x56)
        x58=self.relu13(x57)
        return x58

m = M().eval()
x55 = torch.randn(torch.Size([1, 448, 28, 28]))
start = time.time()
output = m(x55)
end = time.time()
print(end-start)
