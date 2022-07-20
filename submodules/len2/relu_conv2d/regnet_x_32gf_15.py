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
        self.relu15 = ReLU(inplace=True)
        self.conv2d18 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x56):
        x57=self.relu15(x56)
        x58=self.conv2d18(x57)
        return x58

m = M().eval()
x56 = torch.randn(torch.Size([1, 672, 28, 28]))
start = time.time()
output = m(x56)
end = time.time()
print(end-start)
