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
        self.conv2d51 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x151):
        x152=x151.mean((2, 3),keepdim=True)
        x153=self.conv2d51(x152)
        return x153

m = M().eval()
x151 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x151)
end = time.time()
print(end-start)
