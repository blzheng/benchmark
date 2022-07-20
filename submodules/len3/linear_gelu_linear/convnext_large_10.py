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
        self.linear20 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu10 = GELU(approximate='none')
        self.linear21 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x132):
        x133=self.linear20(x132)
        x134=self.gelu10(x133)
        x135=self.linear21(x134)
        return x135

m = M().eval()
x132 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x132)
end = time.time()
print(end-start)
