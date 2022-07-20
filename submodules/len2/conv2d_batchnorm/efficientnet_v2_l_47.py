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
        self.conv2d57 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x191):
        x192=self.conv2d57(x191)
        x193=self.batchnorm2d47(x192)
        return x193

m = M().eval()
x191 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x191)
end = time.time()
print(end-start)
