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

    def forward(self, x204, x194):
        x205=operator.add(x204, x194)
        return x205

m = M().eval()
x204 = torch.randn(torch.Size([1, 768, 14, 14]))
x194 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x204, x194)
end = time.time()
print(end-start)
