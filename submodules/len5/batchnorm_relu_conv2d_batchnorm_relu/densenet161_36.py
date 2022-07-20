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
        self.batchnorm2d75 = BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu75 = ReLU(inplace=True)
        self.conv2d75 = Conv2d(1248, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d76 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu76 = ReLU(inplace=True)

    def forward(self, x267):
        x268=self.batchnorm2d75(x267)
        x269=self.relu75(x268)
        x270=self.conv2d75(x269)
        x271=self.batchnorm2d76(x270)
        x272=self.relu76(x271)
        return x272

m = M().eval()
x267 = torch.randn(torch.Size([1, 1248, 14, 14]))
start = time.time()
output = m(x267)
end = time.time()
print(end-start)
