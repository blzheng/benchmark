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
        self.conv2d138 = Conv2d(200, 1200, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x411, x397):
        x412=operator.add(x411, x397)
        x413=self.conv2d138(x412)
        return x413

m = M().eval()
x411 = torch.randn(torch.Size([1, 200, 14, 14]))
x397 = torch.randn(torch.Size([1, 200, 14, 14]))
start = time.time()
output = m(x411, x397)
end = time.time()
print(end-start)
