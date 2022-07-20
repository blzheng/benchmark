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

    def forward(self, x203, x199):
        x204=self.sigmoid13(x203)
        x205=operator.mul(x204, x199)
        return x205

m = M().eval()
x203 = torch.randn(torch.Size([1, 480, 1, 1]))
x199 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x203, x199)
end = time.time()
print(end-start)
