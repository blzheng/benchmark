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

    def forward(self, x446, x442):
        x447=operator.add(x446, x442)
        return x447

m = M().eval()
x446 = torch.randn(torch.Size([1, 384, 768]))
x442 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x446, x442)
end = time.time()
print(end-start)
