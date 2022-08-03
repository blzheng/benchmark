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

    def forward(self, x219):
        x221=operator.getitem(x219, 1)
        return x221

m = M().eval()
x219 = (1, 96, 14, 14, )
start = time.time()
output = m(x219)
end = time.time()
print(end-start)
