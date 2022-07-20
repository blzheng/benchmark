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
        self.sigmoid22 = Sigmoid()

    def forward(self, x421, x417):
        x422=self.sigmoid22(x421)
        x423=operator.mul(x422, x417)
        return x423

m = M().eval()
x421 = torch.randn(torch.Size([1, 1536, 1, 1]))
x417 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x421, x417)
end = time.time()
print(end-start)
