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
        self.conv2d12 = Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x43, x37):
        x44=operator.add(x43, x37)
        x45=self.conv2d12(x44)
        return x45

m = M().eval()
x43 = torch.randn(torch.Size([1, 48, 56, 56]))
x37 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x43, x37)
end = time.time()
print(end-start)
