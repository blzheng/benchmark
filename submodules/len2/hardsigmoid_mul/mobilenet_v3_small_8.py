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
        self.hardsigmoid8 = Hardsigmoid()

    def forward(self, x143, x139):
        x144=self.hardsigmoid8(x143)
        x145=operator.mul(x144, x139)
        return x145

m = M().eval()
x143 = torch.randn(torch.Size([1, 576, 1, 1]))
x139 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x143, x139)
end = time.time()
print(end-start)