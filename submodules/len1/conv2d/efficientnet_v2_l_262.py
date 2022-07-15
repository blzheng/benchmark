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
        self.conv2d262 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x843):
        x844=self.conv2d262(x843)
        return x844

m = M().eval()
x843 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x843)
end = time.time()
print(end-start)
