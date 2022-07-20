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
        self.conv2d12 = Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x45):
        x46=self.conv2d12(x45)
        x47=self.batchnorm2d12(x46)
        return x47

m = M().eval()
x45 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x45)
end = time.time()
print(end-start)