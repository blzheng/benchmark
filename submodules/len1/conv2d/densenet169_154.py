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
        self.conv2d154 = Conv2d(1440, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x547):
        x548=self.conv2d154(x547)
        return x548

m = M().eval()
x547 = torch.randn(torch.Size([1, 1440, 7, 7]))
start = time.time()
output = m(x547)
end = time.time()
print(end-start)
