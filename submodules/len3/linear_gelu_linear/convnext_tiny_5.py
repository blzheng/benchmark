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
        self.linear10 = Linear(in_features=192, out_features=768, bias=True)
        self.gelu5 = GELU(approximate='none')
        self.linear11 = Linear(in_features=768, out_features=192, bias=True)

    def forward(self, x71):
        x72=self.linear10(x71)
        x73=self.gelu5(x72)
        x74=self.linear11(x73)
        return x74

m = M().eval()
x71 = torch.randn(torch.Size([1, 28, 28, 192]))
start = time.time()
output = m(x71)
end = time.time()
print(end-start)
