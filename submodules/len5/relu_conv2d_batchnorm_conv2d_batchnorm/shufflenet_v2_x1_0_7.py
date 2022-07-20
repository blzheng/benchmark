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
        self.relu17 = ReLU(inplace=True)
        self.conv2d27 = Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116, bias=False)
        self.batchnorm2d27 = BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d28 = Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x167):
        x168=self.relu17(x167)
        x169=self.conv2d27(x168)
        x170=self.batchnorm2d27(x169)
        x171=self.conv2d28(x170)
        x172=self.batchnorm2d28(x171)
        return x172

m = M().eval()
x167 = torch.randn(torch.Size([1, 116, 14, 14]))
start = time.time()
output = m(x167)
end = time.time()
print(end-start)
