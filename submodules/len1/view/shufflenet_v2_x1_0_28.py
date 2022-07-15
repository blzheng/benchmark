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

    def forward(self, x332, x336, x334, x335):
        x337=x330.view(x332, 2, x336, x334, x335)
        return x337

m = M().eval()
x332 = 1
x336 = 232
x334 = 7
x335 = 7
start = time.time()
output = m(x332, x336, x334, x335)
end = time.time()
print(end-start)
