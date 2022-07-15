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

    def forward(self, x84):
        x85=torch.nn.functional.relu(x84,inplace=True)
        return x85

m = M().eval()
x84 = torch.randn(torch.Size([1, 64, 25, 25]))
start = time.time()
output = m(x84)
end = time.time()
print(end-start)
