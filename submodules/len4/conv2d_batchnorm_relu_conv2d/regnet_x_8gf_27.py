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
        self.conv2d43 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)

    def forward(self, x139):
        x140=self.conv2d43(x139)
        x141=self.batchnorm2d43(x140)
        x142=self.relu40(x141)
        x143=self.conv2d44(x142)
        return x143

m = M().eval()
x139 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x139)
end = time.time()
print(end-start)
