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

    def forward(self, x257, x271):
        x272=operator.add(x257, x271)
        return x272

m = M().eval()
x257 = torch.randn(torch.Size([1, 7, 7, 768]))
x271 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x257, x271)
end = time.time()
print(end-start)
