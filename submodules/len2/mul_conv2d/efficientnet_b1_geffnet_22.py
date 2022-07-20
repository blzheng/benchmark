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
        self.conv2d113 = Conv2d(1920, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x328, x333):
        x334=operator.mul(x328, x333)
        x335=self.conv2d113(x334)
        return x335

m = M().eval()
x328 = torch.randn(torch.Size([1, 1920, 7, 7]))
x333 = torch.randn(torch.Size([1, 1920, 1, 1]))
start = time.time()
output = m(x328, x333)
end = time.time()
print(end-start)