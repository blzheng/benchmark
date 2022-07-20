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
        self.conv2d12 = Conv2d(16, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(48, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)

    def forward(self, x34):
        x35=self.conv2d12(x34)
        x36=self.batchnorm2d12(x35)
        x37=self.relu8(x36)
        return x37

m = M().eval()
x34 = torch.randn(torch.Size([1, 16, 56, 56]))
start = time.time()
output = m(x34)
end = time.time()
print(end-start)
