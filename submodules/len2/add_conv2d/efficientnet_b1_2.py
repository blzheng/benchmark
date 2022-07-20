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
        self.conv2d24 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x72, x57):
        x73=operator.add(x72, x57)
        x74=self.conv2d24(x73)
        return x74

m = M().eval()
x72 = torch.randn(torch.Size([1, 24, 56, 56]))
x57 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x72, x57)
end = time.time()
print(end-start)
