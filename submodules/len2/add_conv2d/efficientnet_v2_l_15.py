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
        self.conv2d33 = Conv2d(96, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x116, x110):
        x117=operator.add(x116, x110)
        x118=self.conv2d33(x117)
        return x118

m = M().eval()
x116 = torch.randn(torch.Size([1, 96, 28, 28]))
x110 = torch.randn(torch.Size([1, 96, 28, 28]))
start = time.time()
output = m(x116, x110)
end = time.time()
print(end-start)