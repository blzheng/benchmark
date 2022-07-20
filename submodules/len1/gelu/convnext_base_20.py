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
        self.gelu20 = GELU(approximate='none')

    def forward(self, x243):
        x244=self.gelu20(x243)
        return x244

m = M().eval()
x243 = torch.randn(torch.Size([1, 14, 14, 2048]))
start = time.time()
output = m(x243)
end = time.time()
print(end-start)
