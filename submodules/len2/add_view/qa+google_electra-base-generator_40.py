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

    def forward(self, x453, x451):
        x454=operator.add(x453, (4, 64))
        x455=x451.view(x454)
        return x455

m = M().eval()
x453 = (1, 384, )
x451 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x453, x451)
end = time.time()
print(end-start)
