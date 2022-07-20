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

    def forward(self, x322, x307, x338):
        x323=operator.add(x322, x307)
        x339=operator.add(x338, x323)
        return x339

m = M().eval()
x322 = torch.randn(torch.Size([1, 160, 14, 14]))
x307 = torch.randn(torch.Size([1, 160, 14, 14]))
x338 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x322, x307, x338)
end = time.time()
print(end-start)
