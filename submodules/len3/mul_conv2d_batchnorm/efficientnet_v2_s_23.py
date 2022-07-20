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
        self.conv2d138 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d90 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x438, x433):
        x439=operator.mul(x438, x433)
        x440=self.conv2d138(x439)
        x441=self.batchnorm2d90(x440)
        return x441

m = M().eval()
x438 = torch.randn(torch.Size([1, 1536, 1, 1]))
x433 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x438, x433)
end = time.time()
print(end-start)
