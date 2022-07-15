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
        self.conv2d87 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x273):
        x274=self.conv2d87(x273)
        return x274

m = M().eval()
x273 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x273)
end = time.time()
print(end-start)
