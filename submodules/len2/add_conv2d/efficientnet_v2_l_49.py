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
        self.conv2d218 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x702, x687):
        x703=operator.add(x702, x687)
        x704=self.conv2d218(x703)
        return x704

m = M().eval()
x702 = torch.randn(torch.Size([1, 384, 7, 7]))
x687 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x702, x687)
end = time.time()
print(end-start)
