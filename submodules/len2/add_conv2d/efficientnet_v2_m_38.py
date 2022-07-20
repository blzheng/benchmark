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
        self.conv2d179 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x573, x558):
        x574=operator.add(x573, x558)
        x575=self.conv2d179(x574)
        return x575

m = M().eval()
x573 = torch.randn(torch.Size([1, 304, 7, 7]))
x558 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x573, x558)
end = time.time()
print(end-start)
