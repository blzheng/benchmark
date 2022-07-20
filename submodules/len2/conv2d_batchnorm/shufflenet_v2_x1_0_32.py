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
        self.conv2d32 = Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x209):
        x210=self.conv2d32(x209)
        x211=self.batchnorm2d32(x210)
        return x211

m = M().eval()
x209 = torch.randn(torch.Size([1, 116, 14, 14]))
start = time.time()
output = m(x209)
end = time.time()
print(end-start)
