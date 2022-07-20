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
        self.conv2d8 = Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x18, x23):
        x24=operator.mul(x18, x23)
        x25=self.conv2d8(x24)
        return x25

m = M().eval()
x18 = torch.randn(torch.Size([1, 32, 112, 112]))
x23 = torch.randn(torch.Size([1, 32, 1, 1]))
start = time.time()
output = m(x18, x23)
end = time.time()
print(end-start)
