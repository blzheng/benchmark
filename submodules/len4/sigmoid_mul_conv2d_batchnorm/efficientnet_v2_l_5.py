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
        self.sigmoid5 = Sigmoid()
        self.conv2d62 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x205, x201):
        x206=self.sigmoid5(x205)
        x207=operator.mul(x206, x201)
        x208=self.conv2d62(x207)
        x209=self.batchnorm2d50(x208)
        return x209

m = M().eval()
x205 = torch.randn(torch.Size([1, 768, 1, 1]))
x201 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x205, x201)
end = time.time()
print(end-start)
