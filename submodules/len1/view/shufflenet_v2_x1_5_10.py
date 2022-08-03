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

    def forward(self, x130, x132, x136, x134, x135):
        x137=x130.view(x132, 2, x136, x134, x135)
        return x137

m = M().eval()
x130 = torch.randn(torch.Size([1, 352, 14, 14]))
x132 = 1
x136 = 176
x134 = 14
x135 = 14
start = time.time()
output = m(x130, x132, x136, x134, x135)
end = time.time()
print(end-start)
