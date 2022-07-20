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
        self.conv2d162 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid27 = Sigmoid()

    def forward(self, x519, x516):
        x520=self.conv2d162(x519)
        x521=self.sigmoid27(x520)
        x522=operator.mul(x521, x516)
        return x522

m = M().eval()
x519 = torch.randn(torch.Size([1, 76, 1, 1]))
x516 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x519, x516)
end = time.time()
print(end-start)
