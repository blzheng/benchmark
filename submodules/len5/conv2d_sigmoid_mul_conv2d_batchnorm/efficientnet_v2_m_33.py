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
        self.conv2d192 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid33 = Sigmoid()
        self.conv2d193 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d125 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x615, x612):
        x616=self.conv2d192(x615)
        x617=self.sigmoid33(x616)
        x618=operator.mul(x617, x612)
        x619=self.conv2d193(x618)
        x620=self.batchnorm2d125(x619)
        return x620

m = M().eval()
x615 = torch.randn(torch.Size([1, 76, 1, 1]))
x612 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x615, x612)
end = time.time()
print(end-start)
