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
        self.conv2d89 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x285):
        x286=self.conv2d89(x285)
        x287=self.batchnorm2d61(x286)
        return x287

m = M().eval()
x285 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x285)
end = time.time()
print(end-start)
