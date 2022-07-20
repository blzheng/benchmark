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

    def forward(self, x19):
        x20=torch.permute(x19, [0, 2, 3, 1])
        return x20

m = M().eval()
x19 = torch.randn(torch.Size([1, 128, 56, 56]))
start = time.time()
output = m(x19)
end = time.time()
print(end-start)
