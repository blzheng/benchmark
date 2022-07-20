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
        self.conv2d70 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x212):
        x213=self.conv2d70(x212)
        x214=self.batchnorm2d42(x213)
        return x214

m = M().eval()
x212 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x212)
end = time.time()
print(end-start)
