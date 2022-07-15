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
        self.conv2d60 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x216):
        x217=self.conv2d60(x216)
        return x217

m = M().eval()
x216 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x216)
end = time.time()
print(end-start)
