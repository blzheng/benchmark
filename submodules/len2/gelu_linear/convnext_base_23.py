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
        self.gelu23 = GELU(approximate='none')
        self.linear47 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x276):
        x277=self.gelu23(x276)
        x278=self.linear47(x277)
        return x278

m = M().eval()
x276 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x276)
end = time.time()
print(end-start)
