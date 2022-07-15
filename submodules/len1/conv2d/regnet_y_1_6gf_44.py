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
        self.conv2d44 = Conv2d(120, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x135):
        x138=self.conv2d44(x135)
        return x138

m = M().eval()
x135 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x135)
end = time.time()
print(end-start)
