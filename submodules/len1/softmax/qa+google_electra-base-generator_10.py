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

    def forward(self, x472):
        x473=torch.nn.functional.softmax(x472,dim=-1, _stacklevel=3, dtype=None)
        return x473

m = M().eval()
x472 = torch.randn(torch.Size([1, 4, 384, 384]))
start = time.time()
output = m(x472)
end = time.time()
print(end-start)
