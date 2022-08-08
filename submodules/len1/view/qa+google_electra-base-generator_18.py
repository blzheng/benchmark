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

    def forward(self, x198, x213):
        x214=x198.view(x213)
        return x214

m = M().eval()
x198 = torch.randn(torch.Size([1, 384, 256]))
x213 = (1, 384, 4, 64, )
start = time.time()
output = m(x198, x213)
end = time.time()
print(end-start)
