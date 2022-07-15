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
        self.linear0 = Linear(in_features=1280, out_features=1000, bias=True)

    def forward(self, x545):
        x546=self.linear0(x545)
        return x546

m = M().eval()
x545 = torch.randn(torch.Size([1, 1280]))
start = time.time()
output = m(x545)
end = time.time()
print(end-start)
