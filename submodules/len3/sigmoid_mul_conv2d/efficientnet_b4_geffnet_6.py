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
        self.conv2d33 = Conv2d(192, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x97, x93):
        x98=x97.sigmoid()
        x99=operator.mul(x93, x98)
        x100=self.conv2d33(x99)
        return x100

m = M().eval()
x97 = torch.randn(torch.Size([1, 192, 1, 1]))
x93 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x97, x93)
end = time.time()
print(end-start)
