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
        self.conv2d17 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)

    def forward(self, x54, x48):
        x55=self.conv2d17(x54)
        x56=self.batchnorm2d17(x55)
        x57=operator.add(x56, x48)
        x58=self.relu13(x57)
        return x58

m = M().eval()
x54 = torch.randn(torch.Size([1, 128, 28, 28]))
x48 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x54, x48)
end = time.time()
print(end-start)
