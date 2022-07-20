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
        self.conv2d223 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d147 = BatchNorm2d(2304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x718, x703):
        x719=operator.add(x718, x703)
        x720=self.conv2d223(x719)
        x721=self.batchnorm2d147(x720)
        return x721

m = M().eval()
x718 = torch.randn(torch.Size([1, 384, 7, 7]))
x703 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x718, x703)
end = time.time()
print(end-start)
