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
        self.gelu14 = GELU(approximate='none')
        self.dropout28 = Dropout(p=0.0, inplace=False)

    def forward(self, x358):
        x359=self.gelu14(x358)
        x360=self.dropout28(x359)
        return x360

m = M().eval()
x358 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x358)
end = time.time()
print(end-start)
