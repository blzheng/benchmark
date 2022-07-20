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
        self.conv2d222 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d146 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x715):
        x716=self.conv2d222(x715)
        x717=self.batchnorm2d146(x716)
        return x717

m = M().eval()
x715 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x715)
end = time.time()
print(end-start)
