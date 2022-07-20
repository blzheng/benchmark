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
        self.conv2d244 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x730):
        x731=x730.mean((2, 3),keepdim=True)
        x732=self.conv2d244(x731)
        return x732

m = M().eval()
x730 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x730)
end = time.time()
print(end-start)
