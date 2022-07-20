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
        self.conv2d157 = Conv2d(1200, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x491, x486):
        x492=operator.mul(x491, x486)
        x493=self.conv2d157(x492)
        return x493

m = M().eval()
x491 = torch.randn(torch.Size([1, 1200, 1, 1]))
x486 = torch.randn(torch.Size([1, 1200, 7, 7]))
start = time.time()
output = m(x491, x486)
end = time.time()
print(end-start)
