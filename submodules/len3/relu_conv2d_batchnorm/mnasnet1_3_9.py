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
        self.relu9 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(96, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(56, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x39):
        x40=self.relu9(x39)
        x41=self.conv2d14(x40)
        x42=self.batchnorm2d14(x41)
        return x42

m = M().eval()
x39 = torch.randn(torch.Size([1, 96, 28, 28]))
start = time.time()
output = m(x39)
end = time.time()
print(end-start)
