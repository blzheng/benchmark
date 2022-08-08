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

    def forward(self, x170, x156):
        x171=operator.add(x170, (4, 64))
        x172=x156.view(x171)
        return x172

m = M().eval()
x170 = (1, 384, )
x156 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x170, x156)
end = time.time()
print(end-start)
