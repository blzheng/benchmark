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
        self.linear3 = Linear(in_features=384, out_features=96, bias=True)

    def forward(self, x45):
        x46=self.linear3(x45)
        return x46

m = M().eval()
x45 = torch.randn(torch.Size([1, 56, 56, 384]))
start = time.time()
output = m(x45)
end = time.time()
print(end-start)
