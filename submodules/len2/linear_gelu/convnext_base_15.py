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
        self.linear30 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu15 = GELU(approximate='none')

    def forward(self, x187):
        x188=self.linear30(x187)
        x189=self.gelu15(x188)
        return x189

m = M().eval()
x187 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x187)
end = time.time()
print(end-start)
