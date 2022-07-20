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
        self.conv2d137 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x430, x415):
        x431=operator.add(x430, x415)
        x432=self.conv2d137(x431)
        return x432

m = M().eval()
x430 = torch.randn(torch.Size([1, 160, 14, 14]))
x415 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x430, x415)
end = time.time()
print(end-start)
