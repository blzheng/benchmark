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
        self.hardsigmoid1 = Hardsigmoid()

    def forward(self, x52, x48):
        x53=self.hardsigmoid1(x52)
        x54=operator.mul(x53, x48)
        return x54

m = M().eval()
x52 = torch.randn(torch.Size([1, 120, 1, 1]))
x48 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x52, x48)
end = time.time()
print(end-start)
