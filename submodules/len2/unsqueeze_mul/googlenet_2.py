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

    def forward(self, x9):
        x10=torch.unsqueeze(x9, 1)
        x11=operator.mul(x10, 0.45)
        return x11

m = M().eval()
x9 = torch.randn(torch.Size([1, 224, 224]))
start = time.time()
output = m(x9)
end = time.time()
print(end-start)
