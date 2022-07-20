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
        self.linear22 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu11 = GELU(approximate='none')

    def forward(self, x143):
        x144=self.linear22(x143)
        x145=self.gelu11(x144)
        return x145

m = M().eval()
x143 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x143)
end = time.time()
print(end-start)
