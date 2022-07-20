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

    def forward(self, x72, x57, x88):
        x73=operator.add(x72, x57)
        x89=operator.add(x88, x73)
        return x89

m = M().eval()
x72 = torch.randn(torch.Size([1, 32, 56, 56]))
x57 = torch.randn(torch.Size([1, 32, 56, 56]))
x88 = torch.randn(torch.Size([1, 32, 56, 56]))
start = time.time()
output = m(x72, x57, x88)
end = time.time()
print(end-start)
