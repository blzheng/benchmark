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
        self.conv2d161 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid32 = Sigmoid()

    def forward(self, x503):
        x504=self.conv2d161(x503)
        x505=self.sigmoid32(x504)
        return x505

m = M().eval()
x503 = torch.randn(torch.Size([1, 76, 1, 1]))
start = time.time()
output = m(x503)
end = time.time()
print(end-start)
