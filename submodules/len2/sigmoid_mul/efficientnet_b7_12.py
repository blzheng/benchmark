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
        self.sigmoid12 = Sigmoid()

    def forward(self, x187, x183):
        x188=self.sigmoid12(x187)
        x189=operator.mul(x188, x183)
        return x189

m = M().eval()
x187 = torch.randn(torch.Size([1, 480, 1, 1]))
x183 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x187, x183)
end = time.time()
print(end-start)
