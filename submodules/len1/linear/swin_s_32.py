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
        self.linear32 = Linear(in_features=384, out_features=1536, bias=True)

    def forward(self, x380):
        x381=self.linear32(x380)
        return x381

m = M().eval()
x380 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x380)
end = time.time()
print(end-start)
