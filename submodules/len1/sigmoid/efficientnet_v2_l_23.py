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
        self.sigmoid23 = Sigmoid()

    def forward(self, x491):
        x492=self.sigmoid23(x491)
        return x492

m = M().eval()
x491 = torch.randn(torch.Size([1, 1344, 1, 1]))
start = time.time()
output = m(x491)
end = time.time()
print(end-start)
