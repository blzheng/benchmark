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
        self.gelu4 = GELU(approximate='none')

    def forward(self, x61):
        x62=self.gelu4(x61)
        return x62

m = M().eval()
x61 = torch.randn(torch.Size([1, 28, 28, 1024]))
start = time.time()
output = m(x61)
end = time.time()
print(end-start)
