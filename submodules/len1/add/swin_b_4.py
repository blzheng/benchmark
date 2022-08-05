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

    def forward(self, x57, x71):
        x72=operator.add(x57, x71)
        return x72

m = M().eval()
x57 = torch.randn(torch.Size([1, 28, 28, 256]))
x71 = torch.randn(torch.Size([1, 28, 28, 256]))
start = time.time()
output = m(x57, x71)
end = time.time()
print(end-start)
