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
        self.conv2d153 = Conv2d(1632, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x449, x454):
        x455=operator.mul(x449, x454)
        x456=self.conv2d153(x455)
        return x456

m = M().eval()
x449 = torch.randn(torch.Size([1, 1632, 7, 7]))
x454 = torch.randn(torch.Size([1, 1632, 1, 1]))
start = time.time()
output = m(x449, x454)
end = time.time()
print(end-start)
