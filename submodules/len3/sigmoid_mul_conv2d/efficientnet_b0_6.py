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
        self.sigmoid6 = Sigmoid()
        self.conv2d34 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x98, x94):
        x99=self.sigmoid6(x98)
        x100=operator.mul(x99, x94)
        x101=self.conv2d34(x100)
        return x101

m = M().eval()
x98 = torch.randn(torch.Size([1, 480, 1, 1]))
x94 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x98, x94)
end = time.time()
print(end-start)
