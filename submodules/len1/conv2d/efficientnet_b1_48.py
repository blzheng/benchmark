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
        self.conv2d48 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x145):
        x146=self.conv2d48(x145)
        return x146

m = M().eval()
x145 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x145)
end = time.time()
print(end-start)
