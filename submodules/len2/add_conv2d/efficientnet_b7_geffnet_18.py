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
        self.conv2d112 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x335, x321):
        x336=operator.add(x335, x321)
        x337=self.conv2d112(x336)
        return x337

m = M().eval()
x335 = torch.randn(torch.Size([1, 160, 14, 14]))
x321 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x335, x321)
end = time.time()
print(end-start)
