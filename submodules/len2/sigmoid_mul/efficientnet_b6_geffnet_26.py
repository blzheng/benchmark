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

    def forward(self, x392, x388):
        x393=x392.sigmoid()
        x394=operator.mul(x388, x393)
        return x394

m = M().eval()
x392 = torch.randn(torch.Size([1, 1200, 1, 1]))
x388 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x392, x388)
end = time.time()
print(end-start)
