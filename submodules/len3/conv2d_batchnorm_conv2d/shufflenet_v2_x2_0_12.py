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
        self.conv2d36 = Conv2d(244, 244, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=244, bias=False)
        self.batchnorm2d36 = BatchNorm2d(244, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d37 = Conv2d(244, 244, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x234):
        x235=self.conv2d36(x234)
        x236=self.batchnorm2d36(x235)
        x237=self.conv2d37(x236)
        return x237

m = M().eval()
x234 = torch.randn(torch.Size([1, 244, 14, 14]))
start = time.time()
output = m(x234)
end = time.time()
print(end-start)
