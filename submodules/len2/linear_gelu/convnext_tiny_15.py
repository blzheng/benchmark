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
        self.linear30 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu15 = GELU(approximate='none')

    def forward(self, x193):
        x194=self.linear30(x193)
        x195=self.gelu15(x194)
        return x195

m = M().eval()
x193 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x193)
end = time.time()
print(end-start)
