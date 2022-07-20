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
        self.conv2d26 = Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x74, x60):
        x75=operator.add(x74, x60)
        x76=self.conv2d26(x75)
        return x76

m = M().eval()
x74 = torch.randn(torch.Size([1, 40, 14, 14]))
x60 = torch.randn(torch.Size([1, 40, 14, 14]))
start = time.time()
output = m(x74, x60)
end = time.time()
print(end-start)
