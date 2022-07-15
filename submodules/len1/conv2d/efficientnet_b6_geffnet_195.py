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
        self.conv2d195 = Conv2d(2064, 86, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x583):
        x584=self.conv2d195(x583)
        return x584

m = M().eval()
x583 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x583)
end = time.time()
print(end-start)
