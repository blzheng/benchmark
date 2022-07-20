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
        self.conv2d204 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x610):
        x611=x610.mean((2, 3),keepdim=True)
        x612=self.conv2d204(x611)
        return x612

m = M().eval()
x610 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x610)
end = time.time()
print(end-start)
