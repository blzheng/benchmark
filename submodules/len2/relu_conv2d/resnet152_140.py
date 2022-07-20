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
        self.relu139 = ReLU(inplace=True)
        self.conv2d145 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x479):
        x480=self.relu139(x479)
        x481=self.conv2d145(x480)
        return x481

m = M().eval()
x479 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x479)
end = time.time()
print(end-start)
