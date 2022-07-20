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
        self.conv2d128 = Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d76 = BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x374, x379):
        x380=operator.mul(x374, x379)
        x381=self.conv2d128(x380)
        x382=self.batchnorm2d76(x381)
        return x382

m = M().eval()
x374 = torch.randn(torch.Size([1, 1632, 7, 7]))
x379 = torch.randn(torch.Size([1, 1632, 1, 1]))
start = time.time()
output = m(x374, x379)
end = time.time()
print(end-start)
