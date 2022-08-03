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
        self.linear15 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x144):
        x145=self.linear15(x144)
        return x145

m = M().eval()
x144 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x144)
end = time.time()
print(end-start)
