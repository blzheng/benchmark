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
        self.conv2d36 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x116, x107):
        x117=operator.add(x116, x107)
        x118=self.conv2d36(x117)
        return x118

m = M().eval()
x116 = torch.randn(torch.Size([1, 80, 14, 14]))
x107 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x116, x107)
end = time.time()
print(end-start)
