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
        self.batchnorm2d36 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)
        self.conv2d37 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x119, x112):
        x120=self.batchnorm2d36(x119)
        x121=operator.add(x120, x112)
        x122=self.relu31(x121)
        x123=self.conv2d37(x122)
        x124=self.batchnorm2d37(x123)
        return x124

m = M().eval()
x119 = torch.randn(torch.Size([1, 1024, 28, 28]))
x112 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x119, x112)
end = time.time()
print(end-start)
