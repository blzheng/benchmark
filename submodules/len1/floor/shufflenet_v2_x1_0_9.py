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

    def forward(self, x221):
        x224=operator.floordiv(x221, 2)
        return x224

m = M().eval()
x221 = 232
start = time.time()
output = m(x221)
end = time.time()
print(end-start)
