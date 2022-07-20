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
        self.conv2d89 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        self.batchnorm2d89 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x293):
        x294=self.conv2d89(x293)
        x295=self.batchnorm2d89(x294)
        return x295

m = M().eval()
x293 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x293)
end = time.time()
print(end-start)
