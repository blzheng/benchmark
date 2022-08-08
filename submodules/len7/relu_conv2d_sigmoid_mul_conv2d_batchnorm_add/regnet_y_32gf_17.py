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
        self.relu71 = ReLU()
        self.conv2d92 = Conv2d(348, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()
        self.conv2d93 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d57 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x289, x287, x281):
        x290=self.relu71(x289)
        x291=self.conv2d92(x290)
        x292=self.sigmoid17(x291)
        x293=operator.mul(x292, x287)
        x294=self.conv2d93(x293)
        x295=self.batchnorm2d57(x294)
        x296=operator.add(x281, x295)
        return x296

m = M().eval()
x289 = torch.randn(torch.Size([1, 348, 1, 1]))
x287 = torch.randn(torch.Size([1, 1392, 14, 14]))
x281 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x289, x287, x281)
end = time.time()
print(end-start)
