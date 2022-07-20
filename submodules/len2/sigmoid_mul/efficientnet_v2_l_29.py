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
        self.sigmoid29 = Sigmoid()

    def forward(self, x587, x583):
        x588=self.sigmoid29(x587)
        x589=operator.mul(x588, x583)
        return x589

m = M().eval()
x587 = torch.randn(torch.Size([1, 1344, 1, 1]))
x583 = torch.randn(torch.Size([1, 1344, 7, 7]))
start = time.time()
output = m(x587, x583)
end = time.time()
print(end-start)
