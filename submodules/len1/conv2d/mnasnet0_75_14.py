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
        self.conv2d14 = Conv2d(72, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x40):
        x41=self.conv2d14(x40)
        return x41

m = M().eval()
x40 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x40)
end = time.time()
print(end-start)