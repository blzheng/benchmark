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

    def forward(self, x47):
        x48=x47.mean((2, 3),keepdim=True)
        return x48

m = M().eval()
x47 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x47)
end = time.time()
print(end-start)
