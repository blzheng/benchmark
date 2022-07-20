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
        self.conv2d34 = Conv2d(40, 144, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid5 = Hardsigmoid()

    def forward(self, x98):
        x99=self.conv2d34(x98)
        x100=self.hardsigmoid5(x99)
        return x100

m = M().eval()
x98 = torch.randn(torch.Size([1, 40, 1, 1]))
start = time.time()
output = m(x98)
end = time.time()
print(end-start)
