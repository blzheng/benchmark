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
        self.conv2d68 = Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(768, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x210):
        x211=self.conv2d68(x210)
        x212=self.batchnorm2d40(x211)
        return x212

m = M().eval()
x210 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x210)
end = time.time()
print(end-start)
