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
        self.conv2d27 = Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x99):
        x103=self.conv2d27(x99)
        return x103

m = M().eval()
x99 = torch.randn(torch.Size([1, 288, 25, 25]))
start = time.time()
output = m(x99)
end = time.time()
print(end-start)
