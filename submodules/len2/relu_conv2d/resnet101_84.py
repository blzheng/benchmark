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
        self.relu85 = ReLU(inplace=True)
        self.conv2d89 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x292):
        x293=self.relu85(x292)
        x294=self.conv2d89(x293)
        return x294

m = M().eval()
x292 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x292)
end = time.time()
print(end-start)
