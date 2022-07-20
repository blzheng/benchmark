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
        self.conv2d11 = Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x33, x27):
        x34=self.conv2d11(x33)
        x35=self.batchnorm2d11(x34)
        x36=operator.add(x27, x35)
        return x36

m = M().eval()
x33 = torch.randn(torch.Size([1, 192, 28, 28]))
x27 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x33, x27)
end = time.time()
print(end-start)
