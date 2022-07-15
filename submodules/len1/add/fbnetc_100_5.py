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

    def forward(self, x80, x71):
        x81=operator.add(x80, x71)
        return x81

m = M().eval()
x80 = torch.randn(torch.Size([1, 32, 28, 28]))
x71 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x80, x71)
end = time.time()
print(end-start)
