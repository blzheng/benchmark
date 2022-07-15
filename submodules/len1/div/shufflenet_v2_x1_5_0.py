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

    def forward(self, x21):
        x24=operator.floordiv(x21, 2)
        return x24

m = M().eval()
x21 = 176
start = time.time()
output = m(x21)
end = time.time()
print(end-start)
