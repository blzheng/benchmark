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
        self.conv2d22 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x70, x61):
        x71=operator.add(x70, x61)
        x72=self.conv2d22(x71)
        return x72

m = M().eval()
x70 = torch.randn(torch.Size([1, 32, 28, 28]))
x61 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x70, x61)
end = time.time()
print(end-start)
