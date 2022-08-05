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
        self.conv2d49 = Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid5 = Hardsigmoid()

    def forward(self, x145):
        x146=self.conv2d49(x145)
        x147=self.hardsigmoid5(x146)
        return x147

m = M().eval()
x145 = torch.randn(torch.Size([1, 168, 1, 1]))
start = time.time()
output = m(x145)
end = time.time()
print(end-start)
