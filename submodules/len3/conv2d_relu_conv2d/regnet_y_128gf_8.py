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
        self.conv2d45 = Conv2d(1056, 264, kernel_size=(1, 1), stride=(1, 1))
        self.relu35 = ReLU()
        self.conv2d46 = Conv2d(264, 1056, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x142):
        x143=self.conv2d45(x142)
        x144=self.relu35(x143)
        x145=self.conv2d46(x144)
        return x145

m = M().eval()
x142 = torch.randn(torch.Size([1, 1056, 1, 1]))
start = time.time()
output = m(x142)
end = time.time()
print(end-start)
