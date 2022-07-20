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
        self.conv2d63 = Conv2d(960, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(176, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d64 = Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x206):
        x207=self.conv2d63(x206)
        x208=self.batchnorm2d47(x207)
        x209=self.conv2d64(x208)
        return x209

m = M().eval()
x206 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x206)
end = time.time()
print(end-start)
