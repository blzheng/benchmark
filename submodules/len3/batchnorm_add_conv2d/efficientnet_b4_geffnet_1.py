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
        self.batchnorm2d28 = BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d49 = Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x144, x131):
        x145=self.batchnorm2d28(x144)
        x146=operator.add(x145, x131)
        x147=self.conv2d49(x146)
        return x147

m = M().eval()
x144 = torch.randn(torch.Size([1, 56, 28, 28]))
x131 = torch.randn(torch.Size([1, 56, 28, 28]))
start = time.time()
output = m(x144, x131)
end = time.time()
print(end-start)
