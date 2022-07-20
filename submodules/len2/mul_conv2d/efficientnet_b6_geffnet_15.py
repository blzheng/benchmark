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
        self.conv2d77 = Conv2d(432, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x225, x230):
        x231=operator.mul(x225, x230)
        x232=self.conv2d77(x231)
        return x232

m = M().eval()
x225 = torch.randn(torch.Size([1, 432, 14, 14]))
x230 = torch.randn(torch.Size([1, 432, 1, 1]))
start = time.time()
output = m(x225, x230)
end = time.time()
print(end-start)
