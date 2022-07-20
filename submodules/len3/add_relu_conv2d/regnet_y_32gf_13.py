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
        self.relu56 = ReLU(inplace=True)
        self.conv2d74 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x217, x231):
        x232=operator.add(x217, x231)
        x233=self.relu56(x232)
        x234=self.conv2d74(x233)
        return x234

m = M().eval()
x217 = torch.randn(torch.Size([1, 1392, 14, 14]))
x231 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x217, x231)
end = time.time()
print(end-start)
