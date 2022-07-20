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
        self.conv2d93 = Conv2d(816, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d94 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x274, x270):
        x275=x274.sigmoid()
        x276=operator.mul(x270, x275)
        x277=self.conv2d93(x276)
        x278=self.batchnorm2d55(x277)
        x279=self.conv2d94(x278)
        return x279

m = M().eval()
x274 = torch.randn(torch.Size([1, 816, 1, 1]))
x270 = torch.randn(torch.Size([1, 816, 7, 7]))
start = time.time()
output = m(x274, x270)
end = time.time()
print(end-start)
