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
        self.conv2d64 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x193):
        x194=x193.mean((2, 3),keepdim=True)
        x195=self.conv2d64(x194)
        return x195

m = M().eval()
x193 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x193)
end = time.time()
print(end-start)
