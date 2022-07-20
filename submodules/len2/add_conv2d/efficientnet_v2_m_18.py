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
        self.conv2d74 = Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x239, x224):
        x240=operator.add(x239, x224)
        x241=self.conv2d74(x240)
        return x241

m = M().eval()
x239 = torch.randn(torch.Size([1, 176, 14, 14]))
x224 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x239, x224)
end = time.time()
print(end-start)
