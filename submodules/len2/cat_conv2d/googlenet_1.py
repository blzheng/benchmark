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
        self.conv2d9 = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x27, x33, x39, x43):
        x44=torch.cat([x27, x33, x39, x43], 1)
        x45=self.conv2d9(x44)
        return x45

m = M().eval()
x27 = torch.randn(torch.Size([1, 64, 28, 28]))
x33 = torch.randn(torch.Size([1, 128, 28, 28]))
x39 = torch.randn(torch.Size([1, 32, 28, 28]))
x43 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x27, x33, x39, x43)
end = time.time()
print(end-start)
