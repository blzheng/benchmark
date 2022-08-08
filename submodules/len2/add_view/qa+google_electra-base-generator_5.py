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

    def forward(self, x81, x79):
        x82=operator.add(x81, (4, 64))
        x83=x79.view(x82)
        return x83

m = M().eval()
x81 = (1, 384, )
x79 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x81, x79)
end = time.time()
print(end-start)
