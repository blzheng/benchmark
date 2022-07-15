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

    def forward(self, x784):
        x785=torch.flatten(x784, 1)
        return x785

m = M().eval()
x784 = torch.randn(torch.Size([1, 1280, 1, 1]))
start = time.time()
output = m(x784)
end = time.time()
print(end-start)
