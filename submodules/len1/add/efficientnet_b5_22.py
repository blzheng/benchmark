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

    def forward(self, x445, x430):
        x446=operator.add(x445, x430)
        return x446

m = M().eval()
x445 = torch.randn(torch.Size([1, 304, 7, 7]))
x430 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x445, x430)
end = time.time()
print(end-start)
