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
        self.sigmoid21 = Sigmoid()
        self.conv2d107 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x332, x328):
        x333=self.sigmoid21(x332)
        x334=operator.mul(x333, x328)
        x335=self.conv2d107(x334)
        return x335

m = M().eval()
x332 = torch.randn(torch.Size([1, 864, 1, 1]))
x328 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x332, x328)
end = time.time()
print(end-start)
