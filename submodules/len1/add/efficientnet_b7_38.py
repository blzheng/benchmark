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

    def forward(self, x698, x683):
        x699=operator.add(x698, x683)
        return x699

m = M().eval()
x698 = torch.randn(torch.Size([1, 384, 7, 7]))
x683 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x698, x683)
end = time.time()
print(end-start)
