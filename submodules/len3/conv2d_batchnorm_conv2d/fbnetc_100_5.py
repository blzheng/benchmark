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
        self.conv2d63 = Conv2d(1104, 352, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(352, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d64 = Conv2d(352, 1984, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x205):
        x206=self.conv2d63(x205)
        x207=self.batchnorm2d63(x206)
        x208=self.conv2d64(x207)
        return x208

m = M().eval()
x205 = torch.randn(torch.Size([1, 1104, 7, 7]))
start = time.time()
output = m(x205)
end = time.time()
print(end-start)
