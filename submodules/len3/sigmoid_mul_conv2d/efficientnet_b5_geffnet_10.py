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
        self.conv2d52 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x154, x150):
        x155=x154.sigmoid()
        x156=operator.mul(x150, x155)
        x157=self.conv2d52(x156)
        return x157

m = M().eval()
x154 = torch.randn(torch.Size([1, 384, 1, 1]))
x150 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x154, x150)
end = time.time()
print(end-start)
