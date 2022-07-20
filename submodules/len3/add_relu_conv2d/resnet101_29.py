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
        self.relu88 = ReLU(inplace=True)
        self.conv2d94 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x308, x300):
        x309=operator.add(x308, x300)
        x310=self.relu88(x309)
        x311=self.conv2d94(x310)
        return x311

m = M().eval()
x308 = torch.randn(torch.Size([1, 1024, 14, 14]))
x300 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x308, x300)
end = time.time()
print(end-start)
