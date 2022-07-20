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
        self.conv2d23 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x63, x68):
        x69=operator.mul(x63, x68)
        x70=self.conv2d23(x69)
        return x70

m = M().eval()
x63 = torch.randn(torch.Size([1, 192, 56, 56]))
x68 = torch.randn(torch.Size([1, 192, 1, 1]))
start = time.time()
output = m(x63, x68)
end = time.time()
print(end-start)
