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

    def forward(self, x337, x323):
        x338=operator.add(x337, (12, 64))
        x339=x323.view(x338)
        return x339

m = M().eval()
x337 = (1, 384, )
x323 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x337, x323)
end = time.time()
print(end-start)
