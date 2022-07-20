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

    def forward(self, x250):
        x251=x250.chunk(2,dim=1)
        return x251

m = M().eval()
x250 = torch.randn(torch.Size([1, 232, 14, 14]))
start = time.time()
output = m(x250)
end = time.time()
print(end-start)
