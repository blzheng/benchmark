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
        self.batchnorm2d180 = BatchNorm2d(1600, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu180 = ReLU(inplace=True)
        self.conv2d180 = Conv2d(1600, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d181 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu181 = ReLU(inplace=True)

    def forward(self, x636):
        x637=self.batchnorm2d180(x636)
        x638=self.relu180(x637)
        x639=self.conv2d180(x638)
        x640=self.batchnorm2d181(x639)
        x641=self.relu181(x640)
        return x641

m = M().eval()
x636 = torch.randn(torch.Size([1, 1600, 7, 7]))
start = time.time()
output = m(x636)
end = time.time()
print(end-start)
