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

    def forward(self, x356, x363):
        x364=operator.add(x356, x363)
        return x364

m = M().eval()
x356 = torch.randn(torch.Size([1, 14, 14, 384]))
x363 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x356, x363)
end = time.time()
print(end-start)
