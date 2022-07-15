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

    def forward(self, x171, x161):
        x172=operator.add(x171, x161)
        return x172

m = M().eval()
x171 = torch.randn(torch.Size([1, 512, 14, 14]))
x161 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x171, x161)
end = time.time()
print(end-start)
