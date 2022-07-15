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
        self.conv2d158 = Conv2d(2160, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x561):
        x562=self.conv2d158(x561)
        return x562

m = M().eval()
x561 = torch.randn(torch.Size([1, 2160, 7, 7]))
start = time.time()
output = m(x561)
end = time.time()
print(end-start)
