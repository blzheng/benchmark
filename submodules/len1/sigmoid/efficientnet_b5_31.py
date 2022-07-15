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

    def forward(self, x488):
        x489=self.sigmoid31(x488)
        return x489

m = M().eval()
x488 = torch.randn(torch.Size([1, 1824, 1, 1]))
start = time.time()
output = m(x488)
end = time.time()
print(end-start)
