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

    def forward(self, x394, x379):
        x395=operator.add(x394, x379)
        return x395

m = M().eval()
x394 = torch.randn(torch.Size([1, 256, 7, 7]))
x379 = torch.randn(torch.Size([1, 256, 7, 7]))
start = time.time()
output = m(x394, x379)
end = time.time()
print(end-start)
