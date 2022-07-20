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
        self.batchnorm2d126 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu121 = ReLU(inplace=True)
        self.conv2d127 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x417, x410):
        x418=self.batchnorm2d126(x417)
        x419=operator.add(x418, x410)
        x420=self.relu121(x419)
        x421=self.conv2d127(x420)
        return x421

m = M().eval()
x417 = torch.randn(torch.Size([1, 1024, 14, 14]))
x410 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x417, x410)
end = time.time()
print(end-start)
