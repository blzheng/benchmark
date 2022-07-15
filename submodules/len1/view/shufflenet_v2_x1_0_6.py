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

    def forward(self, x86, x90, x88, x89):
        x91=x84.view(x86, 2, x90, x88, x89)
        return x91

m = M().eval()
x86 = 1
x90 = 58
x88 = 28
x89 = 28
start = time.time()
output = m(x86, x90, x88, x89)
end = time.time()
print(end-start)
