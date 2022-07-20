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
        self.conv2d44 = Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x130, x116):
        x131=operator.add(x130, x116)
        x132=self.conv2d44(x131)
        return x132

m = M().eval()
x130 = torch.randn(torch.Size([1, 56, 28, 28]))
x116 = torch.randn(torch.Size([1, 56, 28, 28]))
start = time.time()
output = m(x130, x116)
end = time.time()
print(end-start)
