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
        self.conv2d6 = Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x24):
        x34=self.conv2d6(x24)
        x35=self.batchnorm2d6(x34)
        return x35

m = M().eval()
x24 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x24)
end = time.time()
print(end-start)
