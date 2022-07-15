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

    def forward(self, x666, x661):
        x667=operator.mul(x666, x661)
        return x667

m = M().eval()
x666 = torch.randn(torch.Size([1, 2304, 1, 1]))
x661 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x666, x661)
end = time.time()
print(end-start)
