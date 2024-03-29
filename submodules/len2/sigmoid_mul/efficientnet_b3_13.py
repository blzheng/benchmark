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

    def forward(self, x207, x203):
        x208=self.sigmoid13(x207)
        x209=operator.mul(x208, x203)
        return x209

m = M().eval()
x207 = torch.randn(torch.Size([1, 576, 1, 1]))
x203 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x207, x203)
end = time.time()
print(end-start)
