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
        self.conv2d72 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x238, x233):
        x239=operator.mul(x238, x233)
        x240=self.conv2d72(x239)
        return x240

m = M().eval()
x238 = torch.randn(torch.Size([1, 768, 1, 1]))
x233 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x238, x233)
end = time.time()
print(end-start)
