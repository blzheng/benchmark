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

    def forward(self, x357, x362):
        x363=operator.mul(x357, x362)
        return x363

m = M().eval()
x357 = torch.randn(torch.Size([1, 960, 14, 14]))
x362 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x357, x362)
end = time.time()
print(end-start)
