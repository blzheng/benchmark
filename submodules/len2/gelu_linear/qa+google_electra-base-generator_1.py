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
        self.linear12 = Linear(in_features=1024, out_features=256, bias=True)

    def forward(self, x108):
        x109=torch._C._nn.gelu(x108)
        x110=self.linear12(x109)
        return x110

m = M().eval()
x108 = torch.randn(torch.Size([1, 384, 1024]))
start = time.time()
output = m(x108)
end = time.time()
print(end-start)
