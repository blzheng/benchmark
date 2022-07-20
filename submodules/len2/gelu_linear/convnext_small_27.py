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
        self.gelu27 = GELU(approximate='none')
        self.linear55 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x320):
        x321=self.gelu27(x320)
        x322=self.linear55(x321)
        return x322

m = M().eval()
x320 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x320)
end = time.time()
print(end-start)