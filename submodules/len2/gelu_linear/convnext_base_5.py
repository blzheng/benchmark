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
        self.gelu5 = GELU(approximate='none')
        self.linear11 = Linear(in_features=1024, out_features=256, bias=True)

    def forward(self, x72):
        x73=self.gelu5(x72)
        x74=self.linear11(x73)
        return x74

m = M().eval()
x72 = torch.randn(torch.Size([1, 28, 28, 1024]))
start = time.time()
output = m(x72)
end = time.time()
print(end-start)
