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

    def forward(self, x196, x198, x202, x200, x201):
        x203=x196.view(x198, 2, x202, x200, x201)
        x204=torch.transpose(x203, 1, 2)
        return x204

m = M().eval()
x196 = torch.randn(torch.Size([1, 352, 14, 14]))
x198 = 1
x202 = 176
x200 = 14
x201 = 14
start = time.time()
output = m(x196, x198, x202, x200, x201)
end = time.time()
print(end-start)
