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
        self.conv2d48 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x155, x146):
        x156=operator.add(x155, x146)
        x157=self.conv2d48(x156)
        return x157

m = M().eval()
x155 = torch.randn(torch.Size([1, 96, 14, 14]))
x146 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x155, x146)
end = time.time()
print(end-start)
