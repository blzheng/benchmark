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
        self.conv2d128 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d86 = BatchNorm2d(176, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x412):
        x413=self.conv2d128(x412)
        x414=self.batchnorm2d86(x413)
        return x414

m = M().eval()
x412 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x412)
end = time.time()
print(end-start)
