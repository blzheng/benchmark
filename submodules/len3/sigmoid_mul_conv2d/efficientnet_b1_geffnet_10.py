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
        self.conv2d53 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x155, x151):
        x156=x155.sigmoid()
        x157=operator.mul(x151, x156)
        x158=self.conv2d53(x157)
        return x158

m = M().eval()
x155 = torch.randn(torch.Size([1, 480, 1, 1]))
x151 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x155, x151)
end = time.time()
print(end-start)
