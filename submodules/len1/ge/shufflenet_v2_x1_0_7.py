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

    def forward(self, x41):
        x43=operator.getitem(x41, 1)
        return x43

m = M().eval()
x41 = (1, 116, 28, 28, )
start = time.time()
output = m(x41)
end = time.time()
print(end-start)