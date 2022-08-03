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
        self.linear23 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x273):
        x274=self.linear23(x273)
        return x274

m = M().eval()
x273 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x273)
end = time.time()
print(end-start)
