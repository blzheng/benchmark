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
        self.conv2d19 = Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)

    def forward(self, x55):
        x56=self.conv2d19(x55)
        x57=self.batchnorm2d15(x56)
        x58=self.relu11(x57)
        return x58

m = M().eval()
x55 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x55)
end = time.time()
print(end-start)
