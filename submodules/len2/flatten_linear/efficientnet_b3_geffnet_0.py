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
        self.linear0 = Linear(in_features=1536, out_features=1000, bias=True)

    def forward(self, x386):
        x387=x386.flatten(1)
        x388=self.linear0(x387)
        return x388

m = M().eval()
x386 = torch.randn(torch.Size([1, 1536, 1, 1]))
start = time.time()
output = m(x386)
end = time.time()
print(end-start)
