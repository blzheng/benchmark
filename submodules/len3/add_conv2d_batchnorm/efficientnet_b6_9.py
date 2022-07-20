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
        self.conv2d63 = Conv2d(72, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(432, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x195, x180):
        x196=operator.add(x195, x180)
        x197=self.conv2d63(x196)
        x198=self.batchnorm2d37(x197)
        return x198

m = M().eval()
x195 = torch.randn(torch.Size([1, 72, 28, 28]))
x180 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x195, x180)
end = time.time()
print(end-start)
