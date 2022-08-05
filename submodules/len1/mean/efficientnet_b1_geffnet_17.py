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

    def forward(self, x254):
        x255=x254.mean((2, 3),keepdim=True)
        return x255

m = M().eval()
x254 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x254)
end = time.time()
print(end-start)
