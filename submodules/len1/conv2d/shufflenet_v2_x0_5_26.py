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
        self.conv2d26 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x165):
        x166=self.conv2d26(x165)
        return x166

m = M().eval()
x165 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x165)
end = time.time()
print(end-start)
