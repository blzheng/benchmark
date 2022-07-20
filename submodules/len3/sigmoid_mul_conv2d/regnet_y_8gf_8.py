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
        self.sigmoid8 = Sigmoid()
        self.conv2d48 = Conv2d(896, 896, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x147, x143):
        x148=self.sigmoid8(x147)
        x149=operator.mul(x148, x143)
        x150=self.conv2d48(x149)
        return x150

m = M().eval()
x147 = torch.randn(torch.Size([1, 896, 1, 1]))
x143 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x147, x143)
end = time.time()
print(end-start)
