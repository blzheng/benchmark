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
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x432):
        x433=torch._C._nn.gelu(x432)
        x434=self.linear6(x433)
        return x434

m = M().eval()
x432 = torch.randn(torch.Size([1, 384, 3072]))
start = time.time()
output = m(x432)
end = time.time()
print(end-start)
