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

    def forward(self, x371, x367):
        x372=self.sigmoid22(x371)
        x373=operator.mul(x372, x367)
        return x373

m = M().eval()
x371 = torch.randn(torch.Size([1, 336, 1, 1]))
x367 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x371, x367)
end = time.time()
print(end-start)
