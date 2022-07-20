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

    def forward(self, x559, x545, x574):
        x560=operator.add(x559, x545)
        x575=operator.add(x574, x560)
        return x575

m = M().eval()
x559 = torch.randn(torch.Size([1, 512, 7, 7]))
x545 = torch.randn(torch.Size([1, 512, 7, 7]))
x574 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x559, x545, x574)
end = time.time()
print(end-start)
