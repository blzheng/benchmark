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

    def forward(self, x74, x83):
        x84=torch.cat((x74, x83),dim=1)
        return x84

m = M().eval()
x74 = torch.randn(torch.Size([1, 122, 28, 28]))
x83 = torch.randn(torch.Size([1, 122, 28, 28]))
start = time.time()
output = m(x74, x83)
end = time.time()
print(end-start)