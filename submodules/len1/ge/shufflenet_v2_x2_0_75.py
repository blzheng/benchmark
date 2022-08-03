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

    def forward(self, x309):
        x311=operator.getitem(x309, 1)
        return x311

m = M().eval()
x309 = (1, 976, 7, 7, )
start = time.time()
output = m(x309)
end = time.time()
print(end-start)
