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

    def forward(self, x27, x19):
        x28=operator.add(x27, x19)
        return x28

m = M().eval()
x27 = torch.randn(torch.Size([1, 24, 56, 56]))
x19 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x27, x19)
end = time.time()
print(end-start)
