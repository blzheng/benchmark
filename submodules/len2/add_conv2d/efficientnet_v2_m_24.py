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
        self.conv2d104 = Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x335, x320):
        x336=operator.add(x335, x320)
        x337=self.conv2d104(x336)
        return x337

m = M().eval()
x335 = torch.randn(torch.Size([1, 176, 14, 14]))
x320 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x335, x320)
end = time.time()
print(end-start)
