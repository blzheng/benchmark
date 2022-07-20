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
        self.conv2d120 = Conv2d(1056, 44, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x358):
        x359=x358.mean((2, 3),keepdim=True)
        x360=self.conv2d120(x359)
        return x360

m = M().eval()
x358 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x358)
end = time.time()
print(end-start)
