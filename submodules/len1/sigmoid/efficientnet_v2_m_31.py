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

    def forward(self, x584):
        x585=self.sigmoid31(x584)
        return x585

m = M().eval()
x584 = torch.randn(torch.Size([1, 1824, 1, 1]))
start = time.time()
output = m(x584)
end = time.time()
print(end-start)
