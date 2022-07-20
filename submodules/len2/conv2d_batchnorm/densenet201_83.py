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
        self.conv2d168 = Conv2d(1408, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d169 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x596):
        x597=self.conv2d168(x596)
        x598=self.batchnorm2d169(x597)
        return x598

m = M().eval()
x596 = torch.randn(torch.Size([1, 1408, 7, 7]))
start = time.time()
output = m(x596)
end = time.time()
print(end-start)
