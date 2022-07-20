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
        self.linear0 = Linear(in_features=2016, out_features=1000, bias=True)

    def forward(self, x284):
        x285=x284.flatten(start_dim=1)
        x286=self.linear0(x285)
        return x286

m = M().eval()
x284 = torch.randn(torch.Size([1, 2016, 1, 1]))
start = time.time()
output = m(x284)
end = time.time()
print(end-start)
