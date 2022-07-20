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
        self.conv2d216 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid36 = Sigmoid()
        self.conv2d217 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x696, x693):
        x697=self.conv2d216(x696)
        x698=self.sigmoid36(x697)
        x699=operator.mul(x698, x693)
        x700=self.conv2d217(x699)
        return x700

m = M().eval()
x696 = torch.randn(torch.Size([1, 96, 1, 1]))
x693 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x696, x693)
end = time.time()
print(end-start)
