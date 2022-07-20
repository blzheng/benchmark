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
        self.relu94 = ReLU(inplace=True)
        self.conv2d94 = Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x336):
        x337=self.relu94(x336)
        x338=self.conv2d94(x337)
        return x338

m = M().eval()
x336 = torch.randn(torch.Size([1, 608, 7, 7]))
start = time.time()
output = m(x336)
end = time.time()
print(end-start)
