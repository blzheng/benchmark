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
        self.conv2d40 = Conv2d(244, 244, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x258):
        x259=self.conv2d40(x258)
        return x259

m = M().eval()
x258 = torch.randn(torch.Size([1, 244, 14, 14]))
start = time.time()
output = m(x258)
end = time.time()
print(end-start)
