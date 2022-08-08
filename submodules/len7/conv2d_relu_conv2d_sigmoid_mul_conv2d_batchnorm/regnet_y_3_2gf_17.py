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
        self.conv2d91 = Conv2d(576, 144, kernel_size=(1, 1), stride=(1, 1))
        self.relu71 = ReLU()
        self.conv2d92 = Conv2d(144, 576, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()
        self.conv2d93 = Conv2d(576, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d57 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x288, x287):
        x289=self.conv2d91(x288)
        x290=self.relu71(x289)
        x291=self.conv2d92(x290)
        x292=self.sigmoid17(x291)
        x293=operator.mul(x292, x287)
        x294=self.conv2d93(x293)
        x295=self.batchnorm2d57(x294)
        return x295

m = M().eval()
x288 = torch.randn(torch.Size([1, 576, 1, 1]))
x287 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x288, x287)
end = time.time()
print(end-start)
