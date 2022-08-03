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
        self.linear16 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x145):
        x146=self.linear16(x145)
        return x146

m = M().eval()
x145 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x145)
end = time.time()
print(end-start)
