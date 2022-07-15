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

    def forward(self, x522, x527):
        x528=operator.mul(x522, x527)
        return x528

m = M().eval()
x522 = torch.randn(torch.Size([1, 2064, 7, 7]))
x527 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x522, x527)
end = time.time()
print(end-start)
