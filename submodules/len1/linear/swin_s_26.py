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
        self.linear26 = Linear(in_features=384, out_features=1536, bias=True)

    def forward(self, x311):
        x312=self.linear26(x311)
        return x312

m = M().eval()
x311 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x311)
end = time.time()
print(end-start)
