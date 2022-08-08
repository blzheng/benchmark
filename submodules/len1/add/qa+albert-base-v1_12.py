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

    def forward(self, x66, x96):
        x97=operator.add(x66, x96)
        return x97

m = M().eval()
x66 = torch.randn(torch.Size([1, 384, 768]))
x96 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x66, x96)
end = time.time()
print(end-start)
