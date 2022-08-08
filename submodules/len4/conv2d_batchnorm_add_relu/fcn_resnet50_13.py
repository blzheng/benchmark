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
        self.conv2d36 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)

    def forward(self, x118, x112):
        x119=self.conv2d36(x118)
        x120=self.batchnorm2d36(x119)
        x121=operator.add(x120, x112)
        x122=self.relu31(x121)
        return x122

m = M().eval()
x118 = torch.randn(torch.Size([1, 256, 28, 28]))
x112 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x118, x112)
end = time.time()
print(end-start)
