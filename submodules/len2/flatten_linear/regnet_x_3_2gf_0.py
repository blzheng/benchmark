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
        self.linear0 = Linear(in_features=1008, out_features=1000, bias=True)

    def forward(self, x262):
        x263=x262.flatten(start_dim=1)
        x264=self.linear0(x263)
        return x264

m = M().eval()
x262 = torch.randn(torch.Size([1, 1008, 1, 1]))
start = time.time()
output = m(x262)
end = time.time()
print(end-start)
