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
        self.conv2d66 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x204, x199):
        x205=operator.mul(x204, x199)
        x206=self.conv2d66(x205)
        return x206

m = M().eval()
x204 = torch.randn(torch.Size([1, 480, 1, 1]))
x199 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x204, x199)
end = time.time()
print(end-start)
