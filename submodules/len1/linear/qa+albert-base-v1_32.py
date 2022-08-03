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
        self.linear2 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x214):
        x216=self.linear2(x214)
        return x216

m = M().eval()
x214 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x214)
end = time.time()
print(end-start)
