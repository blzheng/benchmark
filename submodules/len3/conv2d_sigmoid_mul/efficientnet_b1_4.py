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
        self.conv2d22 = Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()

    def forward(self, x66, x63):
        x67=self.conv2d22(x66)
        x68=self.sigmoid4(x67)
        x69=operator.mul(x68, x63)
        return x69

m = M().eval()
x66 = torch.randn(torch.Size([1, 6, 1, 1]))
x63 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x66, x63)
end = time.time()
print(end-start)
