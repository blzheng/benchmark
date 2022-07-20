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
        self.conv2d87 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid16 = Sigmoid()

    def forward(self, x274):
        x275=self.conv2d87(x274)
        x276=self.sigmoid16(x275)
        return x276

m = M().eval()
x274 = torch.randn(torch.Size([1, 726, 1, 1]))
start = time.time()
output = m(x274)
end = time.time()
print(end-start)
