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

    def forward(self, x164, x162):
        x165=operator.add(x164, (12, 64))
        x166=x162.view(x165)
        return x166

m = M().eval()
x164 = (1, 384, )
x162 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x164, x162)
end = time.time()
print(end-start)
