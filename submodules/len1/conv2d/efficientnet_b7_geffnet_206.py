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
        self.conv2d206 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x616):
        x617=self.conv2d206(x616)
        return x617

m = M().eval()
x616 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x616)
end = time.time()
print(end-start)
