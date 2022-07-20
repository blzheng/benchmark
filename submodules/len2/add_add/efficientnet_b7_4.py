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

    def forward(self, x572, x557, x588):
        x573=operator.add(x572, x557)
        x589=operator.add(x588, x573)
        return x589

m = M().eval()
x572 = torch.randn(torch.Size([1, 224, 14, 14]))
x557 = torch.randn(torch.Size([1, 224, 14, 14]))
x588 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x572, x557, x588)
end = time.time()
print(end-start)
