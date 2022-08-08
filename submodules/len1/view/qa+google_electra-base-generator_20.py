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

    def forward(self, x241, x244):
        x245=x241.view(x244)
        return x245

m = M().eval()
x241 = torch.randn(torch.Size([1, 384, 256]))
x244 = (1, 384, 4, 64, )
start = time.time()
output = m(x241, x244)
end = time.time()
print(end-start)
