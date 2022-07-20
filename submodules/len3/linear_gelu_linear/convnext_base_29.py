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
        self.linear58 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu29 = GELU(approximate='none')
        self.linear59 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x341):
        x342=self.linear58(x341)
        x343=self.gelu29(x342)
        x344=self.linear59(x343)
        return x344

m = M().eval()
x341 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x341)
end = time.time()
print(end-start)
