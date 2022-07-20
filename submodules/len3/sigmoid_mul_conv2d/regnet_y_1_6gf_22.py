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
        self.sigmoid22 = Sigmoid()
        self.conv2d118 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x371, x367):
        x372=self.sigmoid22(x371)
        x373=operator.mul(x372, x367)
        x374=self.conv2d118(x373)
        return x374

m = M().eval()
x371 = torch.randn(torch.Size([1, 336, 1, 1]))
x367 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x371, x367)
end = time.time()
print(end-start)
