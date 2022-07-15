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
        self.sigmoid13 = Sigmoid()

    def forward(self, x227):
        x228=self.sigmoid13(x227)
        return x228

m = M().eval()
x227 = torch.randn(torch.Size([1, 336, 1, 1]))
start = time.time()
output = m(x227)
end = time.time()
print(end-start)
