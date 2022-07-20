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
        self.relu1 = ReLU()
        self.conv2d3 = Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid0 = Hardsigmoid()

    def forward(self, x8):
        x9=self.relu1(x8)
        x10=self.conv2d3(x9)
        x11=self.hardsigmoid0(x10)
        return x11

m = M().eval()
x8 = torch.randn(torch.Size([1, 8, 1, 1]))
start = time.time()
output = m(x8)
end = time.time()
print(end-start)
