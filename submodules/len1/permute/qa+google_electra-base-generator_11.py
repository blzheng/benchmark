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

    def forward(self, x139):
        x140=x139.permute(0, 2, 1, 3)
        return x140

m = M().eval()
x139 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x139)
end = time.time()
print(end-start)
