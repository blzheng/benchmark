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
        self.conv2d32 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x97, x83):
        x98=operator.add(x97, x83)
        x99=self.conv2d32(x98)
        return x99

m = M().eval()
x97 = torch.randn(torch.Size([1, 48, 56, 56]))
x83 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x97, x83)
end = time.time()
print(end-start)
