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
        self.conv2d38 = Conv2d(432, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(144, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x109):
        x110=self.conv2d38(x109)
        x111=self.batchnorm2d38(x110)
        return x111

m = M().eval()
x109 = torch.randn(torch.Size([1, 432, 7, 7]))
start = time.time()
output = m(x109)
end = time.time()
print(end-start)
