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
        self.conv2d83 = Conv2d(144, 864, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x257, x242):
        x258=operator.add(x257, x242)
        x259=self.conv2d83(x258)
        return x259

m = M().eval()
x257 = torch.randn(torch.Size([1, 144, 14, 14]))
x242 = torch.randn(torch.Size([1, 144, 14, 14]))
start = time.time()
output = m(x257, x242)
end = time.time()
print(end-start)
