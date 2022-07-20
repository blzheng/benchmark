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
        self.conv2d21 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x59, x51):
        x60=operator.add(x59, x51)
        x61=self.conv2d21(x60)
        return x61

m = M().eval()
x59 = torch.randn(torch.Size([1, 40, 28, 28]))
x51 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x59, x51)
end = time.time()
print(end-start)
