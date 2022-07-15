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

    def forward(self, x246, x232):
        x247=operator.add(x246, x232)
        return x247

m = M().eval()
x246 = torch.randn(torch.Size([1, 80, 28, 28]))
x232 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x246, x232)
end = time.time()
print(end-start)
