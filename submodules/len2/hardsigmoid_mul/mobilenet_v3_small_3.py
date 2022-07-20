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
        self.hardsigmoid3 = Hardsigmoid()

    def forward(self, x70, x66):
        x71=self.hardsigmoid3(x70)
        x72=operator.mul(x71, x66)
        return x72

m = M().eval()
x70 = torch.randn(torch.Size([1, 240, 1, 1]))
x66 = torch.randn(torch.Size([1, 240, 14, 14]))
start = time.time()
output = m(x70, x66)
end = time.time()
print(end-start)
