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

    def forward(self, x513):
        x514=torch.nn.functional.softmax(x513,dim=-1, _stacklevel=3, dtype=None)
        return x514

m = M().eval()
x513 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x513)
end = time.time()
print(end-start)
