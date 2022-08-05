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

    def forward(self, x539):
        x540=x539.contiguous()
        return x540

m = M().eval()
x539 = torch.randn(torch.Size([32, 49, 49]))
start = time.time()
output = m(x539)
end = time.time()
print(end-start)
