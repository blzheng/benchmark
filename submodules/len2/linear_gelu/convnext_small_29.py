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
        self.linear58 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu29 = GELU(approximate='none')

    def forward(self, x341):
        x342=self.linear58(x341)
        x343=self.gelu29(x342)
        return x343

m = M().eval()
x341 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x341)
end = time.time()
print(end-start)
