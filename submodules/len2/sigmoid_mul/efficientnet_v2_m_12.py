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
        self.sigmoid12 = Sigmoid()

    def forward(self, x282, x278):
        x283=self.sigmoid12(x282)
        x284=operator.mul(x283, x278)
        return x284

m = M().eval()
x282 = torch.randn(torch.Size([1, 1056, 1, 1]))
x278 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x282, x278)
end = time.time()
print(end-start)
