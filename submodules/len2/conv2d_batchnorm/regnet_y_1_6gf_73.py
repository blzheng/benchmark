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
        self.conv2d119 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x377):
        x378=self.conv2d119(x377)
        x379=self.batchnorm2d73(x378)
        return x379

m = M().eval()
x377 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x377)
end = time.time()
print(end-start)
