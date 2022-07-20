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
        self.conv2d102 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()

    def forward(self, x329):
        x330=self.conv2d102(x329)
        x331=self.sigmoid15(x330)
        return x331

m = M().eval()
x329 = torch.randn(torch.Size([1, 44, 1, 1]))
start = time.time()
output = m(x329)
end = time.time()
print(end-start)
