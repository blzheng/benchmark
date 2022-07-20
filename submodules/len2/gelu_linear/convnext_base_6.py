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
        self.gelu6 = GELU(approximate='none')
        self.linear13 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x89):
        x90=self.gelu6(x89)
        x91=self.linear13(x90)
        return x91

m = M().eval()
x89 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x89)
end = time.time()
print(end-start)
