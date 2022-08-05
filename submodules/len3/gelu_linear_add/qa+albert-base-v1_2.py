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

    def forward(self, x136, x135):
        x137=torch._C._nn.gelu(x136)
        x138=self.linear6(x137)
        x139=operator.add(x138, x135)
        return x139

m = M().eval()
x136 = torch.randn(torch.Size([1, 384, 3072]))
x135 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x136, x135)
end = time.time()
print(end-start)
