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
        self.conv2d52 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x167):
        x168=self.conv2d52(x167)
        return x168

m = M().eval()
x167 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x167)
end = time.time()
print(end-start)
