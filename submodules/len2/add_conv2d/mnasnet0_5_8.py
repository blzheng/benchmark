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
        self.conv2d45 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x128, x120):
        x129=operator.add(x128, x120)
        x130=self.conv2d45(x129)
        return x130

m = M().eval()
x128 = torch.randn(torch.Size([1, 96, 7, 7]))
x120 = torch.randn(torch.Size([1, 96, 7, 7]))
start = time.time()
output = m(x128, x120)
end = time.time()
print(end-start)
