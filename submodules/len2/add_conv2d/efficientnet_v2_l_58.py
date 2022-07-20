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
        self.conv2d263 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x846, x831):
        x847=operator.add(x846, x831)
        x848=self.conv2d263(x847)
        return x848

m = M().eval()
x846 = torch.randn(torch.Size([1, 384, 7, 7]))
x831 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x846, x831)
end = time.time()
print(end-start)
