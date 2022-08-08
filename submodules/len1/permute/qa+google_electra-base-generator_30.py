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

    def forward(self, x340):
        x341=x340.permute(0, 2, 1, 3)
        return x341

m = M().eval()
x340 = torch.randn(torch.Size([1, 384, 4, 64]))
start = time.time()
output = m(x340)
end = time.time()
print(end-start)
