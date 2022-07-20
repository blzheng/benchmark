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
        self.conv2d222 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x663, x649):
        x664=operator.add(x663, x649)
        x665=self.conv2d222(x664)
        return x665

m = M().eval()
x663 = torch.randn(torch.Size([1, 384, 7, 7]))
x649 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x663, x649)
end = time.time()
print(end-start)
