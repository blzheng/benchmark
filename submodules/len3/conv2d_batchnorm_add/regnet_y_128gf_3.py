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
        self.conv2d12 = Conv2d(528, 1056, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d8 = BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x37, x53):
        x38=self.conv2d12(x37)
        x39=self.batchnorm2d8(x38)
        x54=operator.add(x39, x53)
        return x54

m = M().eval()
x37 = torch.randn(torch.Size([1, 528, 56, 56]))
x53 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x37, x53)
end = time.time()
print(end-start)
