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
        self.conv2d39 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x103, x111):
        x112=operator.add(x103, x111)
        x113=self.conv2d39(x112)
        return x113

m = M().eval()
x103 = torch.randn(torch.Size([1, 96, 14, 14]))
x111 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x103, x111)
end = time.time()
print(end-start)
