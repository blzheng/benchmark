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
        self.conv2d212 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d126 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x666, x651):
        x667=operator.add(x666, x651)
        x668=self.conv2d212(x667)
        x669=self.batchnorm2d126(x668)
        return x669

m = M().eval()
x666 = torch.randn(torch.Size([1, 384, 7, 7]))
x651 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x666, x651)
end = time.time()
print(end-start)
