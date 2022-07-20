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
        self.conv2d145 = Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x472):
        x473=self.conv2d145(x472)
        return x473

m = M().eval()
x472 = torch.randn(torch.Size([1, 1344, 1, 1]))
start = time.time()
output = m(x472)
end = time.time()
print(end-start)