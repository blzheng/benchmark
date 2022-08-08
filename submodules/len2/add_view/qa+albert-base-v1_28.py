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

    def forward(self, x372, x364):
        x373=operator.add(x372, (12, 64))
        x374=x364.view(x373)
        return x374

m = M().eval()
x372 = (1, 384, )
x364 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x372, x364)
end = time.time()
print(end-start)
