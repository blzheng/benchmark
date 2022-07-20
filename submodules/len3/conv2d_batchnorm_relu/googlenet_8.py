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
        self.conv2d8 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x40):
        x41=self.conv2d8(x40)
        x42=self.batchnorm2d8(x41)
        x43=torch.nn.functional.relu(x42,inplace=True)
        return x43

m = M().eval()
x40 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x40)
end = time.time()
print(end-start)
