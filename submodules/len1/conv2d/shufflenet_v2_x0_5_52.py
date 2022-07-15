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
        self.conv2d52 = Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x343):
        x344=self.conv2d52(x343)
        return x344

m = M().eval()
x343 = torch.randn(torch.Size([1, 96, 7, 7]))
start = time.time()
output = m(x343)
end = time.time()
print(end-start)
