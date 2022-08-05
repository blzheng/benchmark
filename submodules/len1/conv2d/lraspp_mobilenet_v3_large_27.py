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
        self.conv2d27 = Conv2d(80, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x78):
        x79=self.conv2d27(x78)
        return x79

m = M().eval()
x78 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x78)
end = time.time()
print(end-start)
