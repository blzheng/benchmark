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
        self.batchnorm2d8 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)
        self.conv2d18 = Conv2d(120, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x38, x53):
        x39=self.batchnorm2d8(x38)
        x54=operator.add(x39, x53)
        x55=self.relu12(x54)
        x56=self.conv2d18(x55)
        return x56

m = M().eval()
x38 = torch.randn(torch.Size([1, 120, 28, 28]))
x53 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x38, x53)
end = time.time()
print(end-start)
