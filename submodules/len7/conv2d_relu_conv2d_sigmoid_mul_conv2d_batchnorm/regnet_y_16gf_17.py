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
        self.conv2d92 = Conv2d(3024, 308, kernel_size=(1, 1), stride=(1, 1))
        self.relu71 = ReLU()
        self.conv2d93 = Conv2d(308, 3024, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()
        self.conv2d94 = Conv2d(3024, 3024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(3024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x290, x289):
        x291=self.conv2d92(x290)
        x292=self.relu71(x291)
        x293=self.conv2d93(x292)
        x294=self.sigmoid17(x293)
        x295=operator.mul(x294, x289)
        x296=self.conv2d94(x295)
        x297=self.batchnorm2d58(x296)
        return x297

m = M().eval()
x290 = torch.randn(torch.Size([1, 3024, 1, 1]))
x289 = torch.randn(torch.Size([1, 3024, 7, 7]))
start = time.time()
output = m(x290, x289)
end = time.time()
print(end-start)
