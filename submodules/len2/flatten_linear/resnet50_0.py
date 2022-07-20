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
        self.linear0 = Linear(in_features=2048, out_features=1000, bias=True)

    def forward(self, x173):
        x174=torch.flatten(x173, 1)
        x175=self.linear0(x174)
        return x175

m = M().eval()
x173 = torch.randn(torch.Size([1, 2048, 1, 1]))
start = time.time()
output = m(x173)
end = time.time()
print(end-start)
