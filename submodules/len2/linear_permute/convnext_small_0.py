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
        self.linear1 = Linear(in_features=384, out_features=96, bias=True)

    def forward(self, x12):
        x13=self.linear1(x12)
        x14=torch.permute(x13, [0, 3, 1, 2])
        return x14

m = M().eval()
x12 = torch.randn(torch.Size([1, 56, 56, 384]))
start = time.time()
output = m(x12)
end = time.time()
print(end-start)
