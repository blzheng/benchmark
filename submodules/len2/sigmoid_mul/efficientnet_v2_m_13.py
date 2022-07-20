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

    def forward(self, x298, x294):
        x299=self.sigmoid13(x298)
        x300=operator.mul(x299, x294)
        return x300

m = M().eval()
x298 = torch.randn(torch.Size([1, 1056, 1, 1]))
x294 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x298, x294)
end = time.time()
print(end-start)
