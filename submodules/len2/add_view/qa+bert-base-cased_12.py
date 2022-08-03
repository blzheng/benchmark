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

    def forward(self, x158, x156):
        x159=operator.add(x158, (12, 64))
        x160=x156.view(x159)
        return x160

m = M().eval()
x158 = (1, 384, )
x156 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x158, x156)
end = time.time()
print(end-start)
