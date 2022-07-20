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
        self.conv2d133 = Conv2d(84, 888, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid25 = Sigmoid()

    def forward(self, x420, x417):
        x421=self.conv2d133(x420)
        x422=self.sigmoid25(x421)
        x423=operator.mul(x422, x417)
        return x423

m = M().eval()
x420 = torch.randn(torch.Size([1, 84, 1, 1]))
x417 = torch.randn(torch.Size([1, 888, 7, 7]))
start = time.time()
output = m(x420, x417)
end = time.time()
print(end-start)
