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
        self.conv2d96 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d56 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x299):
        x300=self.conv2d96(x299)
        x301=self.batchnorm2d56(x300)
        return x301

m = M().eval()
x299 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x299)
end = time.time()
print(end-start)