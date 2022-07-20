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
        self.conv2d104 = Conv2d(576, 1512, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x329):
        x330=self.conv2d104(x329)
        return x330

m = M().eval()
x329 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x329)
end = time.time()
print(end-start)