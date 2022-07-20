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
        self.sigmoid5 = Sigmoid()
        self.conv2d48 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x153, x149):
        x154=self.sigmoid5(x153)
        x155=operator.mul(x154, x149)
        x156=self.conv2d48(x155)
        return x156

m = M().eval()
x153 = torch.randn(torch.Size([1, 512, 1, 1]))
x149 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x153, x149)
end = time.time()
print(end-start)
