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
        self.linear6 = Linear(in_features=192, out_features=768, bias=True)
        self.gelu3 = GELU(approximate='none')

    def forward(self, x49):
        x50=self.linear6(x49)
        x51=self.gelu3(x50)
        return x51

m = M().eval()
x49 = torch.randn(torch.Size([1, 28, 28, 192]))
start = time.time()
output = m(x49)
end = time.time()
print(end-start)
