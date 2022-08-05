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

    def forward(self, x111, x125):
        x126=operator.add(x111, x125)
        return x126

m = M().eval()
x111 = torch.randn(torch.Size([1, 14, 14, 384]))
x125 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x111, x125)
end = time.time()
print(end-start)
