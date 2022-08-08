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

    def forward(self, x505, x491):
        x506=operator.add(x505, (12, 64))
        x507=x491.view(x506)
        return x507

m = M().eval()
x505 = (1, 384, )
x491 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x505, x491)
end = time.time()
print(end-start)
