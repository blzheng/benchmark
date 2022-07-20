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
        self.conv2d173 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x515, x501):
        x516=operator.add(x515, x501)
        x517=self.conv2d173(x516)
        return x517

m = M().eval()
x515 = torch.randn(torch.Size([1, 304, 7, 7]))
x501 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x515, x501)
end = time.time()
print(end-start)
