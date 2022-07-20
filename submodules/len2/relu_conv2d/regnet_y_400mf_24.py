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
        self.relu32 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(208, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x136):
        x137=self.relu32(x136)
        x138=self.conv2d44(x137)
        return x138

m = M().eval()
x136 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x136)
end = time.time()
print(end-start)
