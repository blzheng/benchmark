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

    def forward(self, x332, x330):
        x333=operator.add(x332, (12, 64))
        x334=x330.view(x333)
        return x334

m = M().eval()
x332 = (1, 384, )
x330 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x332, x330)
end = time.time()
print(end-start)
