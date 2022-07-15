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

    def forward(self, x926, x911):
        x927=operator.add(x926, x911)
        return x927

m = M().eval()
x926 = torch.randn(torch.Size([1, 384, 7, 7]))
x911 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x926, x911)
end = time.time()
print(end-start)
