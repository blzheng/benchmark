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
        self.conv2d147 = Conv2d(1200, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x459, x454):
        x460=operator.mul(x459, x454)
        x461=self.conv2d147(x460)
        return x461

m = M().eval()
x459 = torch.randn(torch.Size([1, 1200, 1, 1]))
x454 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x459, x454)
end = time.time()
print(end-start)
