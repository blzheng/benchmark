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
        self.conv2d52 = Conv2d(52, 208, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()
        self.conv2d53 = Conv2d(208, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)
        self.conv2d54 = Conv2d(208, 440, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x162, x159, x153):
        x163=self.conv2d52(x162)
        x164=self.sigmoid9(x163)
        x165=operator.mul(x164, x159)
        x166=self.conv2d53(x165)
        x167=self.batchnorm2d33(x166)
        x168=operator.add(x153, x167)
        x169=self.relu40(x168)
        x170=self.conv2d54(x169)
        return x170

m = M().eval()
x162 = torch.randn(torch.Size([1, 52, 1, 1]))
x159 = torch.randn(torch.Size([1, 208, 14, 14]))
x153 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x162, x159, x153)
end = time.time()
print(end-start)
