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
        self.linear32 = Linear(in_features=512, out_features=2048, bias=True)

    def forward(self, x198):
        x199=self.linear32(x198)
        return x199

m = M().eval()
x198 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x198)
end = time.time()
print(end-start)
