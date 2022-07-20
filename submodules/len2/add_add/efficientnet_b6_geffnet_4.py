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

    def forward(self, x441, x427, x456):
        x442=operator.add(x441, x427)
        x457=operator.add(x456, x442)
        return x457

m = M().eval()
x441 = torch.randn(torch.Size([1, 200, 14, 14]))
x427 = torch.randn(torch.Size([1, 200, 14, 14]))
x456 = torch.randn(torch.Size([1, 200, 14, 14]))
start = time.time()
output = m(x441, x427, x456)
end = time.time()
print(end-start)
