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

    def forward(self, x74):
        x75=operator.add(x74, (12, 64))
        return x75

m = M().eval()
x74 = (1, 384, )
start = time.time()
output = m(x74)
end = time.time()
print(end-start)
