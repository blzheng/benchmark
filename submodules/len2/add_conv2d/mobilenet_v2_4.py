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
        self.conv2d27 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x68, x76):
        x77=operator.add(x68, x76)
        x78=self.conv2d27(x77)
        return x78

m = M().eval()
x68 = torch.randn(torch.Size([1, 64, 14, 14]))
x76 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x68, x76)
end = time.time()
print(end-start)
