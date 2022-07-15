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
        self.conv2d8 = Conv2d(232, 232, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x24):
        x25=self.conv2d8(x24)
        return x25

m = M().eval()
x24 = torch.randn(torch.Size([1, 232, 56, 56]))
start = time.time()
output = m(x24)
end = time.time()
print(end-start)
