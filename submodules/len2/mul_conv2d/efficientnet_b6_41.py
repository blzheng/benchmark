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
        self.conv2d207 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x649, x644):
        x650=operator.mul(x649, x644)
        x651=self.conv2d207(x650)
        return x651

m = M().eval()
x649 = torch.randn(torch.Size([1, 2064, 1, 1]))
x644 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x649, x644)
end = time.time()
print(end-start)
