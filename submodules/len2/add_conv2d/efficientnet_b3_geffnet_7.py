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
        self.conv2d59 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x174, x160):
        x175=operator.add(x174, x160)
        x176=self.conv2d59(x175)
        return x176

m = M().eval()
x174 = torch.randn(torch.Size([1, 96, 14, 14]))
x160 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x174, x160)
end = time.time()
print(end-start)
