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

    def forward(self, x164, x173):
        x174=torch.cat((x164, x173),dim=1)
        return x174

m = M().eval()
x164 = torch.randn(torch.Size([1, 176, 14, 14]))
x173 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x164, x173)
end = time.time()
print(end-start)
