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
        self.conv2d5 = Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d5 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x15):
        x16=self.conv2d5(x15)
        x17=self.batchnorm2d5(x16)
        return x17

m = M().eval()
x15 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x15)
end = time.time()
print(end-start)
