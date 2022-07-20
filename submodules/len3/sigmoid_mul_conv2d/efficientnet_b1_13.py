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
        self.sigmoid13 = Sigmoid()
        self.conv2d68 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x205, x201):
        x206=self.sigmoid13(x205)
        x207=operator.mul(x206, x201)
        x208=self.conv2d68(x207)
        return x208

m = M().eval()
x205 = torch.randn(torch.Size([1, 672, 1, 1]))
x201 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x205, x201)
end = time.time()
print(end-start)
