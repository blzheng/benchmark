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

    def forward(self, x42, x46, x44, x45):
        x47=x40.view(x42, 2, x46, x44, x45)
        return x47

m = M().eval()
x42 = 1
x46 = 88
x44 = 28
x45 = 28
start = time.time()
output = m(x42, x46, x44, x45)
end = time.time()
print(end-start)
