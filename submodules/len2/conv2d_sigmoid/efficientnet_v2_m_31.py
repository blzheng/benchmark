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
        self.conv2d182 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid31 = Sigmoid()

    def forward(self, x583):
        x584=self.conv2d182(x583)
        x585=self.sigmoid31(x584)
        return x585

m = M().eval()
x583 = torch.randn(torch.Size([1, 76, 1, 1]))
start = time.time()
output = m(x583)
end = time.time()
print(end-start)
