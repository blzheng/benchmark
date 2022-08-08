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

    def forward(self, x489, x485):
        x490=operator.add(x489, x485)
        return x490

m = M().eval()
x489 = torch.randn(torch.Size([1, 384, 256]))
x485 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x489, x485)
end = time.time()
print(end-start)
