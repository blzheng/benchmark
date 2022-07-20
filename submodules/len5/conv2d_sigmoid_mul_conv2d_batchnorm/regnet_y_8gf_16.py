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
        self.conv2d88 = Conv2d(224, 2016, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid16 = Sigmoid()
        self.conv2d89 = Conv2d(2016, 2016, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(2016, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x276, x273):
        x277=self.conv2d88(x276)
        x278=self.sigmoid16(x277)
        x279=operator.mul(x278, x273)
        x280=self.conv2d89(x279)
        x281=self.batchnorm2d55(x280)
        return x281

m = M().eval()
x276 = torch.randn(torch.Size([1, 224, 1, 1]))
x273 = torch.randn(torch.Size([1, 2016, 7, 7]))
start = time.time()
output = m(x276, x273)
end = time.time()
print(end-start)
