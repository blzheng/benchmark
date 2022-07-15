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
        self.conv2d267 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x859):
        x860=self.conv2d267(x859)
        return x860

m = M().eval()
x859 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x859)
end = time.time()
print(end-start)
