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

    def forward(self, x347, x337):
        x348=operator.add(x347, x337)
        return x348

m = M().eval()
x347 = torch.randn(torch.Size([1, 768, 14, 14]))
x337 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x347, x337)
end = time.time()
print(end-start)
