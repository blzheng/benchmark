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
        self.conv2d236 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d140 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x700, x705):
        x706=operator.mul(x700, x705)
        x707=self.conv2d236(x706)
        x708=self.batchnorm2d140(x707)
        return x708

m = M().eval()
x700 = torch.randn(torch.Size([1, 2304, 7, 7]))
x705 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x700, x705)
end = time.time()
print(end-start)
