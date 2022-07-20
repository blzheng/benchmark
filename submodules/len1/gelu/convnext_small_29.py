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
        self.gelu29 = GELU(approximate='none')

    def forward(self, x342):
        x343=self.gelu29(x342)
        return x343

m = M().eval()
x342 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x342)
end = time.time()
print(end-start)
