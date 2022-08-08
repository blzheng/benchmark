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

    def forward(self, x399, x429):
        x430=operator.add(x399, x429)
        return x430

m = M().eval()
x399 = torch.randn(torch.Size([1, 384, 768]))
x429 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x399, x429)
end = time.time()
print(end-start)
