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
        self.gelu18 = GELU(approximate='none')

    def forward(self, x221):
        x222=self.gelu18(x221)
        return x222

m = M().eval()
x221 = torch.randn(torch.Size([1, 14, 14, 3072]))
start = time.time()
output = m(x221)
end = time.time()
print(end-start)
