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
        self.relu39 = ReLU()
        self.conv2d52 = Conv2d(348, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()
        self.conv2d53 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x161, x159, x153):
        x162=self.relu39(x161)
        x163=self.conv2d52(x162)
        x164=self.sigmoid9(x163)
        x165=operator.mul(x164, x159)
        x166=self.conv2d53(x165)
        x167=self.batchnorm2d33(x166)
        x168=operator.add(x153, x167)
        return x168

m = M().eval()
x161 = torch.randn(torch.Size([1, 348, 1, 1]))
x159 = torch.randn(torch.Size([1, 1392, 14, 14]))
x153 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x161, x159, x153)
end = time.time()
print(end-start)
