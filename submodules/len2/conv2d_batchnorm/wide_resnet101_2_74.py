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
        self.conv2d74 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d74 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x243):
        x244=self.conv2d74(x243)
        x245=self.batchnorm2d74(x244)
        return x245

m = M().eval()
x243 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x243)
end = time.time()
print(end-start)
