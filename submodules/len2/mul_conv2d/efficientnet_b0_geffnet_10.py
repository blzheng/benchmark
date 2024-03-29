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
        self.conv2d54 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x152, x157):
        x158=operator.mul(x152, x157)
        x159=self.conv2d54(x158)
        return x159

m = M().eval()
x152 = torch.randn(torch.Size([1, 672, 14, 14]))
x157 = torch.randn(torch.Size([1, 672, 1, 1]))
start = time.time()
output = m(x152, x157)
end = time.time()
print(end-start)
