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
        self.gelu7 = GELU(approximate='none')

    def forward(self, x100):
        x101=self.gelu7(x100)
        return x101

m = M().eval()
x100 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x100)
end = time.time()
print(end-start)
