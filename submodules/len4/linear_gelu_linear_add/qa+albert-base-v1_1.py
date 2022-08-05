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
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x98, x98):
        x99=self.linear5(x98)
        x100=torch._C._nn.gelu(x99)
        x101=self.linear6(x100)
        x102=operator.add(x101, x98)
        return x102

m = M().eval()
x98 = torch.randn(torch.Size([1, 384, 768]))
x98 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x98, x98)
end = time.time()
print(end-start)
