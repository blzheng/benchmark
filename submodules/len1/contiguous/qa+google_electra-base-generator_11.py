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

    def forward(self, x518):
        x519=x518.contiguous()
        return x519

m = M().eval()
x518 = torch.randn(torch.Size([1, 384, 4, 64]))
start = time.time()
output = m(x518)
end = time.time()
print(end-start)
