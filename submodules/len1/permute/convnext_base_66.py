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

    def forward(self, x350):
        x351=torch.permute(x350, [0, 2, 3, 1])
        return x351

m = M().eval()
x350 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x350)
end = time.time()
print(end-start)