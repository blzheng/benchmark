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
        self.hardsigmoid0 = Hardsigmoid()

    def forward(self, x36, x32):
        x37=self.hardsigmoid0(x36)
        x38=operator.mul(x37, x32)
        return x38

m = M().eval()
x36 = torch.randn(torch.Size([1, 72, 1, 1]))
x32 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x36, x32)
end = time.time()
print(end-start)
