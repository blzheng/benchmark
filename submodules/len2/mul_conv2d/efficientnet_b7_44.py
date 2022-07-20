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
        self.conv2d221 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x694, x689):
        x695=operator.mul(x694, x689)
        x696=self.conv2d221(x695)
        return x696

m = M().eval()
x694 = torch.randn(torch.Size([1, 2304, 1, 1]))
x689 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x694, x689)
end = time.time()
print(end-start)
