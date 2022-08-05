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

    def forward(self, x97, x89):
        x98=operator.add(x97, x89)
        return x98

m = M().eval()
x97 = torch.randn(torch.Size([1, 80, 14, 14]))
x89 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x97, x89)
end = time.time()
print(end-start)
