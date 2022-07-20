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

    def forward(self, x525, x521):
        x526=x525.sigmoid()
        x527=operator.mul(x521, x526)
        return x527

m = M().eval()
x525 = torch.randn(torch.Size([1, 1344, 1, 1]))
x521 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x525, x521)
end = time.time()
print(end-start)
