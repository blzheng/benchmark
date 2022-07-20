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
        self.sigmoid37 = Sigmoid()
        self.conv2d187 = Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x582, x578):
        x583=self.sigmoid37(x582)
        x584=operator.mul(x583, x578)
        x585=self.conv2d187(x584)
        return x585

m = M().eval()
x582 = torch.randn(torch.Size([1, 3072, 1, 1]))
x578 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x582, x578)
end = time.time()
print(end-start)
