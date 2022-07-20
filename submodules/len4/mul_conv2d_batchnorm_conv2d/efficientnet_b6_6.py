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
        self.conv2d212 = Conv2d(2064, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d126 = BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d213 = Conv2d(576, 3456, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x665, x660):
        x666=operator.mul(x665, x660)
        x667=self.conv2d212(x666)
        x668=self.batchnorm2d126(x667)
        x669=self.conv2d213(x668)
        return x669

m = M().eval()
x665 = torch.randn(torch.Size([1, 2064, 1, 1]))
x660 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x665, x660)
end = time.time()
print(end-start)
