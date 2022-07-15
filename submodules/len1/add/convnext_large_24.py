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

    def forward(self, x292, x282):
        x293=operator.add(x292, x282)
        return x293

m = M().eval()
x292 = torch.randn(torch.Size([1, 768, 14, 14]))
x282 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x292, x282)
end = time.time()
print(end-start)
