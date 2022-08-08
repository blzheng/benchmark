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
        self.conv2d2 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x9):
        x10=self.conv2d2(x9)
        x11=self.batchnorm2d2(x10)
        return x11

m = M().eval()
x9 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x9)
end = time.time()
print(end-start)
