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

    def forward(self, x252, x237):
        x253=operator.add(x252, x237)
        return x253

m = M().eval()
x252 = torch.randn(torch.Size([1, 160, 14, 14]))
x237 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x252, x237)
end = time.time()
print(end-start)
