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

    def forward(self, x390):
        x391=x390.permute(0, 2, 1, 3)
        return x391

m = M().eval()
x390 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x390)
end = time.time()
print(end-start)
