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

    def forward(self, x290, x288):
        x291=operator.add(x290, (12, 64))
        x292=x288.view(x291)
        return x292

m = M().eval()
x290 = (1, 384, )
x288 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x290, x288)
end = time.time()
print(end-start)
