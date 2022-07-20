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
        self.relu48 = ReLU(inplace=True)
        self.conv2d64 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x200):
        x201=self.relu48(x200)
        x202=self.conv2d64(x201)
        return x202

m = M().eval()
x200 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x200)
end = time.time()
print(end-start)
