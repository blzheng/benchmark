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
        self.gelu16 = GELU(approximate='none')

    def forward(self, x199):
        x200=self.gelu16(x199)
        return x200

m = M().eval()
x199 = torch.randn(torch.Size([1, 14, 14, 3072]))
start = time.time()
output = m(x199)
end = time.time()
print(end-start)
