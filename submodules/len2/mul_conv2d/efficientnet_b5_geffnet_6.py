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
        self.conv2d32 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x91, x96):
        x97=operator.mul(x91, x96)
        x98=self.conv2d32(x97)
        return x98

m = M().eval()
x91 = torch.randn(torch.Size([1, 240, 56, 56]))
x96 = torch.randn(torch.Size([1, 240, 1, 1]))
start = time.time()
output = m(x91, x96)
end = time.time()
print(end-start)
