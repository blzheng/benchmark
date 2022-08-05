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

    def forward(self, x533, x547):
        x548=operator.add(x533, x547)
        return x548

m = M().eval()
x533 = torch.randn(torch.Size([1, 7, 7, 768]))
x547 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x533, x547)
end = time.time()
print(end-start)
