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
        self.conv2d16 = Conv2d(160, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x61):
        x62=self.conv2d16(x61)
        return x62

m = M().eval()
x61 = torch.randn(torch.Size([1, 160, 28, 28]))
start = time.time()
output = m(x61)
end = time.time()
print(end-start)