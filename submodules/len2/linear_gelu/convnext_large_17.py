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
        self.linear34 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu17 = GELU(approximate='none')

    def forward(self, x209):
        x210=self.linear34(x209)
        x211=self.gelu17(x210)
        return x211

m = M().eval()
x209 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x209)
end = time.time()
print(end-start)
