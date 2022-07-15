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
        self.sigmoid17 = Sigmoid()

    def forward(self, x267):
        x268=self.sigmoid17(x267)
        return x268

m = M().eval()
x267 = torch.randn(torch.Size([1, 1152, 1, 1]))
start = time.time()
output = m(x267)
end = time.time()
print(end-start)
