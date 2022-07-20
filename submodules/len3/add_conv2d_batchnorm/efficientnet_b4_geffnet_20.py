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
        self.conv2d134 = Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d80 = BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x397, x383):
        x398=operator.add(x397, x383)
        x399=self.conv2d134(x398)
        x400=self.batchnorm2d80(x399)
        return x400

m = M().eval()
x397 = torch.randn(torch.Size([1, 272, 7, 7]))
x383 = torch.randn(torch.Size([1, 272, 7, 7]))
start = time.time()
output = m(x397, x383)
end = time.time()
print(end-start)
