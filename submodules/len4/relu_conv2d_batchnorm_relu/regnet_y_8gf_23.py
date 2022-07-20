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
        self.relu52 = ReLU(inplace=True)
        self.conv2d69 = Conv2d(896, 896, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu53 = ReLU(inplace=True)

    def forward(self, x216):
        x217=self.relu52(x216)
        x218=self.conv2d69(x217)
        x219=self.batchnorm2d43(x218)
        x220=self.relu53(x219)
        return x220

m = M().eval()
x216 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x216)
end = time.time()
print(end-start)
