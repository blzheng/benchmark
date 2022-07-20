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
        self.conv2d11 = Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x34):
        x35=self.conv2d11(x34)
        return x35

m = M().eval()
x34 = torch.randn(torch.Size([1, 96, 1, 1]))
start = time.time()
output = m(x34)
end = time.time()
print(end-start)