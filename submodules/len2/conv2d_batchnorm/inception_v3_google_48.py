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
        self.conv2d48 = Conv2d(160, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
        self.batchnorm2d48 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x169):
        x170=self.conv2d48(x169)
        x171=self.batchnorm2d48(x170)
        return x171

m = M().eval()
x169 = torch.randn(torch.Size([1, 160, 12, 12]))
start = time.time()
output = m(x169)
end = time.time()
print(end-start)
