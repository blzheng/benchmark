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

    def forward(self, x793, x789):
        x794=x793.sigmoid()
        x795=operator.mul(x789, x794)
        return x795

m = M().eval()
x793 = torch.randn(torch.Size([1, 3840, 1, 1]))
x789 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x793, x789)
end = time.time()
print(end-start)
