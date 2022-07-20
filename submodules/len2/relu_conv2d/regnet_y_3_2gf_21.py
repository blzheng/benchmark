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
        self.relu28 = ReLU(inplace=True)
        self.conv2d38 = Conv2d(216, 576, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x118):
        x119=self.relu28(x118)
        x120=self.conv2d38(x119)
        return x120

m = M().eval()
x118 = torch.randn(torch.Size([1, 216, 28, 28]))
start = time.time()
output = m(x118)
end = time.time()
print(end-start)