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
        self.conv2d63 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()
        self.conv2d64 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x189, x186):
        x190=self.conv2d63(x189)
        x191=self.sigmoid12(x190)
        x192=operator.mul(x191, x186)
        x193=self.conv2d64(x192)
        return x193

m = M().eval()
x189 = torch.randn(torch.Size([1, 48, 1, 1]))
x186 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x189, x186)
end = time.time()
print(end-start)
