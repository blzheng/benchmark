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
        self.relu41 = ReLU(inplace=True)
        self.conv2d45 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x144):
        x145=self.relu41(x144)
        x146=self.conv2d45(x145)
        x147=self.batchnorm2d45(x146)
        return x147

m = M().eval()
x144 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x144)
end = time.time()
print(end-start)
