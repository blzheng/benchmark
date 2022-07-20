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
        self.batchnorm2d51 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu46 = ReLU(inplace=True)
        self.conv2d52 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x167, x160):
        x168=self.batchnorm2d51(x167)
        x169=operator.add(x168, x160)
        x170=self.relu46(x169)
        x171=self.conv2d52(x170)
        x172=self.batchnorm2d52(x171)
        return x172

m = M().eval()
x167 = torch.randn(torch.Size([1, 1024, 14, 14]))
x160 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x167, x160)
end = time.time()
print(end-start)
