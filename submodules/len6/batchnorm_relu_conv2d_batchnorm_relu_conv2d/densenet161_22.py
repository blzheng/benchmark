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
        self.batchnorm2d47 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu47 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu48 = ReLU(inplace=True)
        self.conv2d48 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x169):
        x170=self.batchnorm2d47(x169)
        x171=self.relu47(x170)
        x172=self.conv2d47(x171)
        x173=self.batchnorm2d48(x172)
        x174=self.relu48(x173)
        x175=self.conv2d48(x174)
        return x175

m = M().eval()
x169 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x169)
end = time.time()
print(end-start)
