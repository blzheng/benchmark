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
        self.conv2d197 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x618):
        x619=self.conv2d197(x618)
        return x619

m = M().eval()
x618 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x618)
end = time.time()
print(end-start)
