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
        self.sigmoid40 = Sigmoid()

    def forward(self, x761):
        x762=self.sigmoid40(x761)
        return x762

m = M().eval()
x761 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x761)
end = time.time()
print(end-start)
