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

    def forward(self, x29):
        x31=operator.getitem(x29, 1)
        return x31

m = M().eval()
x29 = (torch.randn((torch.Size([1, 24, 28, 28]), torch.randn(torch.Size([1, 24, 28, 28]), )
start = time.time()
output = m(x29)
end = time.time()
print(end-start)
