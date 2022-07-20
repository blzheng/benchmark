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
        self.conv2d11 = Conv2d(48, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x39):
        x40=self.conv2d11(x39)
        x41=self.batchnorm2d11(x40)
        return x41

m = M().eval()
x39 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x39)
end = time.time()
print(end-start)
