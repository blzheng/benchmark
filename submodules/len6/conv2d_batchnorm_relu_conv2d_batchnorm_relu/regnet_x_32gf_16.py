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
        self.conv2d52 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu49 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(1344, 1344, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
        self.batchnorm2d53 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu50 = ReLU(inplace=True)

    def forward(self, x169):
        x170=self.conv2d52(x169)
        x171=self.batchnorm2d52(x170)
        x172=self.relu49(x171)
        x173=self.conv2d53(x172)
        x174=self.batchnorm2d53(x173)
        x175=self.relu50(x174)
        return x175

m = M().eval()
x169 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x169)
end = time.time()
print(end-start)
