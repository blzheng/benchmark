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

    def forward(self, x93, x86, x88, x89):
        x94=x93.view(x86, -1, x88, x89)
        return x94

m = M().eval()
x93 = torch.randn(torch.Size([1, 122, 2, 28, 28]))
x86 = 1
x88 = 28
x89 = 28
start = time.time()
output = m(x93, x86, x88, x89)
end = time.time()
print(end-start)
