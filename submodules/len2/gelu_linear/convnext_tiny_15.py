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
        self.gelu15 = GELU(approximate='none')
        self.linear31 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x194):
        x195=self.gelu15(x194)
        x196=self.linear31(x195)
        return x196

m = M().eval()
x194 = torch.randn(torch.Size([1, 7, 7, 3072]))
start = time.time()
output = m(x194)
end = time.time()
print(end-start)
