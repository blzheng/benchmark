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

    def forward(self, x200, x198):
        x201=operator.add(x200, (12, 64))
        x202=x198.view(x201)
        return x202

m = M().eval()
x200 = (1, 384, )
x198 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x200, x198)
end = time.time()
print(end-start)
