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
        self.sigmoid14 = Sigmoid()
        self.conv2d79 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x245, x241):
        x246=self.sigmoid14(x245)
        x247=operator.mul(x246, x241)
        x248=self.conv2d79(x247)
        return x248

m = M().eval()
x245 = torch.randn(torch.Size([1, 440, 1, 1]))
x241 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x245, x241)
end = time.time()
print(end-start)
