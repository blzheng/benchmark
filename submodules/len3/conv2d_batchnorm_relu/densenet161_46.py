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
        self.conv2d93 = Conv2d(1680, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d94 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu94 = ReLU(inplace=True)

    def forward(self, x332):
        x333=self.conv2d93(x332)
        x334=self.batchnorm2d94(x333)
        x335=self.relu94(x334)
        return x335

m = M().eval()
x332 = torch.randn(torch.Size([1, 1680, 14, 14]))
start = time.time()
output = m(x332)
end = time.time()
print(end-start)
