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
        self.conv2d45 = Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=42, bias=False)
        self.batchnorm2d45 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu41 = ReLU(inplace=True)
        self.conv2d46 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x144):
        x145=self.conv2d45(x144)
        x146=self.batchnorm2d45(x145)
        x147=self.relu41(x146)
        x148=self.conv2d46(x147)
        return x148

m = M().eval()
x144 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x144)
end = time.time()
print(end-start)
