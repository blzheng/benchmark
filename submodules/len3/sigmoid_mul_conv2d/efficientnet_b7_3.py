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
        self.sigmoid3 = Sigmoid()
        self.conv2d16 = Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x47, x43):
        x48=self.sigmoid3(x47)
        x49=operator.mul(x48, x43)
        x50=self.conv2d16(x49)
        return x50

m = M().eval()
x47 = torch.randn(torch.Size([1, 32, 1, 1]))
x43 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x47, x43)
end = time.time()
print(end-start)
