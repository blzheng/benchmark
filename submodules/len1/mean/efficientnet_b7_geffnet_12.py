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

    def forward(self, x178):
        x179=x178.mean((2, 3),keepdim=True)
        return x179

m = M().eval()
x178 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x178)
end = time.time()
print(end-start)
