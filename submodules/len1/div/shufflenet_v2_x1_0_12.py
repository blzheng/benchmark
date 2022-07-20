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

    def forward(self, x289):
        x292=operator.floordiv(x289, 2)
        return x292

m = M().eval()
x289 = 464
start = time.time()
output = m(x289)
end = time.time()
print(end-start)