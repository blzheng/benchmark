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
        self.sigmoid31 = Sigmoid()

    def forward(self, x490, x486):
        x491=self.sigmoid31(x490)
        x492=operator.mul(x491, x486)
        return x492

m = M().eval()
x490 = torch.randn(torch.Size([1, 1200, 1, 1]))
x486 = torch.randn(torch.Size([1, 1200, 7, 7]))
start = time.time()
output = m(x490, x486)
end = time.time()
print(end-start)
