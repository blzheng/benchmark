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

    def forward(self, x742, x737):
        x743=operator.mul(x742, x737)
        return x743

m = M().eval()
x742 = torch.randn(torch.Size([1, 2304, 1, 1]))
x737 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x742, x737)
end = time.time()
print(end-start)
