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
        self.conv2d22 = Conv2d(120, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x65, x61):
        x66=self.sigmoid3(x65)
        x67=operator.mul(x66, x61)
        x68=self.conv2d22(x67)
        return x68

m = M().eval()
x65 = torch.randn(torch.Size([1, 120, 1, 1]))
x61 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x65, x61)
end = time.time()
print(end-start)
