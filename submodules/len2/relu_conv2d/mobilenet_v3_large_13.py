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
        self.relu16 = ReLU()
        self.conv2d49 = Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x142):
        x143=self.relu16(x142)
        x144=self.conv2d49(x143)
        return x144

m = M().eval()
x142 = torch.randn(torch.Size([1, 168, 1, 1]))
start = time.time()
output = m(x142)
end = time.time()
print(end-start)
