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

    def forward(self, x252, x247):
        x253=operator.mul(x252, x247)
        return x253

m = M().eval()
x252 = torch.randn(torch.Size([1, 480, 1, 1]))
x247 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x252, x247)
end = time.time()
print(end-start)
