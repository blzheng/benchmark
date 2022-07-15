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

    def forward(self, x148, x154, x160, x164):
        x165=torch.cat([x148, x154, x160, x164], 1)
        return x165

m = M().eval()
x148 = torch.randn(torch.Size([1, 256, 14, 14]))
x154 = torch.randn(torch.Size([1, 320, 14, 14]))
x160 = torch.randn(torch.Size([1, 128, 14, 14]))
x164 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x148, x154, x160, x164)
end = time.time()
print(end-start)
