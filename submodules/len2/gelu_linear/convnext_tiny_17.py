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
        self.gelu17 = GELU(approximate='none')
        self.linear35 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x216):
        x217=self.gelu17(x216)
        x218=self.linear35(x217)
        return x218

m = M().eval()
x216 = torch.randn(torch.Size([1, 7, 7, 3072]))
start = time.time()
output = m(x216)
end = time.time()
print(end-start)
