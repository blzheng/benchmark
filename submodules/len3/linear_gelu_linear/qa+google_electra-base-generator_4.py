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
        self.linear29 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear30 = Linear(in_features=1024, out_features=256, bias=True)

    def forward(self, x233):
        x234=self.linear29(x233)
        x235=torch._C._nn.gelu(x234)
        x236=self.linear30(x235)
        return x236

m = M().eval()
x233 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x233)
end = time.time()
print(end-start)
