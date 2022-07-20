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
        self.conv2d64 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu61 = ReLU(inplace=True)

    def forward(self, x209):
        x210=self.conv2d64(x209)
        x211=self.batchnorm2d64(x210)
        x212=self.relu61(x211)
        return x212

m = M().eval()
x209 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x209)
end = time.time()
print(end-start)
