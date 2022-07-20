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
        self.conv2d43 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)

    def forward(self, x139):
        x140=self.conv2d43(x139)
        x141=self.batchnorm2d43(x140)
        x142=self.relu29(x141)
        return x142

m = M().eval()
x139 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x139)
end = time.time()
print(end-start)
