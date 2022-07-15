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

    def forward(self, x276, x271):
        x277=operator.mul(x276, x271)
        return x277

m = M().eval()
x276 = torch.randn(torch.Size([1, 2904, 1, 1]))
x271 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x276, x271)
end = time.time()
print(end-start)
