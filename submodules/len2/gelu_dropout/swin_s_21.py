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
        self.gelu21 = GELU(approximate='none')
        self.dropout42 = Dropout(p=0.0, inplace=False)

    def forward(self, x519):
        x520=self.gelu21(x519)
        x521=self.dropout42(x520)
        return x521

m = M().eval()
x519 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x519)
end = time.time()
print(end-start)
