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
        self.conv2d231 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x685, x690):
        x691=operator.mul(x685, x690)
        x692=self.conv2d231(x691)
        return x692

m = M().eval()
x685 = torch.randn(torch.Size([1, 2304, 7, 7]))
x690 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x685, x690)
end = time.time()
print(end-start)
